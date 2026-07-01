import torch
from typing import List, Optional, Sequence, Tuple
from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    CenterSpatialCropd,
    Lambdad,
    Transposed,
    MapTransform,
    RandSpatialCropSamplesd,
    Resized,
    SpatialCropd,
    RandFlipd,
    RandAffined
)
import os
import shutil


def _make_contiguous(x):
    return x.contiguous()


def _build_volume_transforms(size):
    return Compose([
        LoadImaged(keys=["input", "output"]),
        EnsureChannelFirstd(keys=["input", "output"]),
        ExtractSpacingd(keys=["input"], init_sizes=512, out_sizes=size),
        Transposed(keys=["input", "output"], indices=[0, 3, 1, 2]),
        Lambdad(keys=["input", "output"], func=_make_contiguous),
        CenterSpatialCropd(keys=["input", "output"], roi_size=(-1, 512, 512)),
        EnsureTyped(keys=["input", "output"], track_meta=False),
        Resized(
            keys=["input", "output"],
            spatial_size=(-1, size, size),
            mode=("trilinear", "nearest"),
        ),
    ])

class ExtractSpacingd(MapTransform):
    """
    Custom transform to pull the pixdim out of the NIfTI header
    and save it as a new key in the data dictionary.
    """
    def __init__(self, keys, init_sizes, out_sizes, meta_key_postfix="meta_dict"):
        super().__init__(keys)
        self.init_sizes = init_sizes
        self.out_sizes = out_sizes
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)
        img = d["input"]
        spacing = torch.tensor(img.meta["pixdim"][1:4])
        spacing[:2] *= self.init_sizes / self.out_sizes
        d[f"pixdim"] = spacing
        return d

class VerticalPositionEmbedding(MapTransform):
    """
    Custom transform to add a vertical position embedding to the input data.
    The embedding is a 2D array where each row corresponds to a slice and contains
    the normalized vertical position of that slice in the volume.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]
        pixdim_z = d["pixdim"][-1]
        num_slices = img.shape[1]
        vertical_positions = torch.arange(num_slices).float() / (num_slices - 1) - 0.5
        vertical_positions = vertical_positions.to(img.device, dtype=img.dtype)
        vertical_positions = vertical_positions.view(
            1, num_slices, 1, 1
            ).expand(
            -1, img.shape[1], img.shape[2], img.shape[3]
            ) * pixdim_z.view(-1, 1, 1, 1)
        d["input"] = torch.cat((img, vertical_positions), dim=0)
        return d
    
class RandomSliceDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 base_dataset, 
                 channels, 
                 size, 
                 num_slices=1,
                random_flip=False,
                random_shift=False):
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.base_dataset = base_dataset
        self.channels = channels
        self.patch_transform = Compose([
            VerticalPositionEmbedding(keys=["input"]),  # Add vertical position embedding to the input data
            RandSpatialCropSamplesd(
                keys=["input", "output"],
                roi_size=(channels, size, size), 
                num_samples=num_slices,         
                random_size=False,
            ),
            RandFlipd(
                keys=["input", "output"],
                prob=0.5 if self.random_flip else 0.0,
                spatial_axis=1
            ),
            RandAffined(
                keys=["input", "output"],
                prob=0.5 if self.random_shift else 0.0,
                translate_range=(0, 10, 10),
                padding_mode="border"
            )
            
    ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        volume = self.base_dataset[idx]
        samples = self.patch_transform(volume)
        if not isinstance(samples, list):
            samples = [samples]

        fixed_samples = []
        for sample in samples:
            sample = dict(sample)
            if "case_id" not in sample and "case_id" in volume:
                sample["case_id"] = volume["case_id"]
            sample["case_idx"] = torch.tensor(idx, dtype=torch.long)
            depth = sample["input"].shape[1]
            if depth < self.channels:
                pad = self.channels - depth
                sample["input"] = torch.nn.functional.pad(
                    sample["input"], (0, 0, 0, 0, 0, pad), value=-1.
                ).contiguous()
                sample["output"] = torch.nn.functional.pad(
                    sample["output"], (0, 0, 0, 0, 0, pad), value=-1.
                ).contiguous()
            fixed_samples.append(sample)

        return fixed_samples

class NeighborLoader:

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        channels: int, # Number of slices per chunk
        neighbors: int = 2, # Number of neighboring slices to include on each side
        num_slices: int = 1, # Number of slices per chunk
        device: str = "cuda",
        size: int = 128,
        coarse_size: int = 32,
        include_neighbor_inputs: bool = False,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.channels = channels
        self.neighbors = neighbors
        self.num_slices = num_slices
        self.device = device
        self.size = size
        self.coarse_size = coarse_size
        self.include_neighbor_inputs = include_neighbor_inputs

        self._volume_transform = _build_volume_transforms(size=self.size)
        self._full_inputs: List[torch.Tensor] = []
        self._coarse_inputs: List[torch.Tensor] = []
        self._full_outputs: List[torch.Tensor] = []
        self._case_ids: List[str] = []
        self._case_id_to_index = {}
        self._build_index()

    def _build_index(self):
        case_ids = sorted([file[:9] for file in os.listdir(self.image_dir) if file.endswith(".nii.gz")])
        for case_id in case_ids:
            entry = {
                "input": os.path.join(self.image_dir, f"{case_id}_0000.nii.gz"),
                "output": os.path.join(self.label_dir, f"{case_id}_0001.nii.gz"),
            }
            volume = self._volume_transform(entry)
            inp = volume["input"].detach().to(dtype=torch.float32, device="cpu")
            out = volume["output"].detach().to(dtype=torch.float32, device="cpu")

            coarse_inp = torch.nn.functional.interpolate(
                inp.unsqueeze(0),
                size=(inp.shape[1], self.coarse_size, self.coarse_size),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

            self._full_inputs.append(inp)
            self._coarse_inputs.append(coarse_inp)
            self._full_outputs.append(out)
            self._case_ids.append(case_id)
            self._case_id_to_index[case_id] = len(self._case_ids) - 1

    @staticmethod
    def _normalize_ignore_case_id(ignore_case_id, batch_size: int) -> List[Optional[str]]:
        if ignore_case_id is None:
            return [None] * batch_size
        if isinstance(ignore_case_id, str):
            return [ignore_case_id] * batch_size
        if isinstance(ignore_case_id, bytes):
            return [ignore_case_id.decode("utf-8")] * batch_size
        if isinstance(ignore_case_id, Sequence):
            values = [str(v) if v is not None else None for v in ignore_case_id]
            if len(values) == batch_size:
                return values
            if len(values) == 1:
                return values * batch_size
            raise ValueError("ignore_case_id sequence length does not match batch size")
        return [str(ignore_case_id)] * batch_size

    @staticmethod
    def _extract_depth_window(volume: torch.Tensor, start: int, depth: int) -> torch.Tensor:
        # volume shape: [1, D, H, W]
        d_total = volume.shape[1]
        if d_total >= depth:
            start = max(0, min(start, d_total - depth))
            return volume[:, start:start + depth]

        pad_after = depth - d_total
        padded = torch.nn.functional.pad(volume, (0, 0, 0, 0, 0, pad_after), value=-1.0)
        return padded[:, :depth]

    def _best_candidates(
        self,
        query_coarse: torch.Tensor,
        k: int,
        ignore_case_idx: Optional[int],
    ) -> List[Tuple[float, int, int]]:
        # query_coarse shape: [1, D, Hc, Wc]
        q_depth = int(query_coarse.shape[1])
        best_per_case: List[Tuple[float, int, int]] = []

        for case_idx, coarse_volume in enumerate(self._coarse_inputs):
            if ignore_case_idx is not None and case_idx == ignore_case_idx:
                continue

            d_total = int(coarse_volume.shape[1])
            if d_total <= 0:
                continue

            if d_total >= q_depth:
                starts = range(0, d_total - q_depth + 1)
            else:
                starts = [0]

            best_candidate: Optional[Tuple[float, int, int]] = None
            for start in starts:
                cand = self._extract_depth_window(coarse_volume, start, q_depth)
                dist = torch.mean((cand - query_coarse) ** 2).item()
                candidate = (dist, case_idx, start)
                if best_candidate is None or dist < best_candidate[0]:
                    best_candidate = candidate

            if best_candidate is not None:
                best_per_case.append(best_candidate)

        best_per_case.sort(key=lambda x: x[0])
        return best_per_case[:k]

    def get_neighbors(self, x, ignore_case_id=None, k: Optional[int] = None):
        """
        Find k nearest neighbors for each sample in a batch and return full-resolution
        output patches matching query depth/height/width.

        Args:
            x: Input tensor shaped [B, C, D, H, W] or [B, D, H, W].
            ignore_case_id: Case ID (or per-batch IDs) to exclude from search.
            k: Number of nearest neighbors. Defaults to self.neighbors.

        Returns:
            torch.Tensor shaped [B, k, D, H, W].
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError("Expected input shape [B, C, D, H, W] or [B, D, H, W]")

        batch_size, _, q_depth, q_h, q_w = x.shape
        k = self.neighbors if k is None else int(k)
        if k <= 0:
            raise ValueError("k must be > 0")

        ignore_case_ids = self._normalize_ignore_case_id(ignore_case_id, batch_size)

        query_for_match = x[:, :1].detach().to(device="cpu", dtype=torch.float32)
        query_coarse = torch.nn.functional.interpolate(
            query_for_match,
            size=(q_depth, self.coarse_size, self.coarse_size),
            mode="trilinear",
            align_corners=False,
        )

        neighbors_batch: List[torch.Tensor] = []
        for b in range(batch_size):
            ignore_case_idx = None
            if ignore_case_ids[b] is not None:
                ignore_case_idx = self._case_id_to_index.get(ignore_case_ids[b], None)

            best = self._best_candidates(
                query_coarse=query_coarse[b],
                k=k,
                ignore_case_idx=ignore_case_idx,
            )

            out_neighbors: List[torch.Tensor] = []
            for _, case_idx, start in best:
                in_vol = self._full_inputs[case_idx]
                out_vol = self._full_outputs[case_idx]
                in_patch = self._extract_depth_window(in_vol, start, q_depth).unsqueeze(0)
                out_patch = self._extract_depth_window(out_vol, start, q_depth).unsqueeze(0)
                if in_patch.shape[-2:] != (q_h, q_w):
                    in_patch = torch.nn.functional.interpolate(
                        in_patch,
                        size=(q_depth, q_h, q_w),
                        mode="trilinear",
                        align_corners=False,
                    )
                if out_patch.shape[-2:] != (q_h, q_w):
                    out_patch = torch.nn.functional.interpolate(
                        out_patch,
                        size=(q_depth, q_h, q_w),
                        mode="trilinear",
                        align_corners=False,
                    )

                if self.include_neighbor_inputs:
                    neighbor_patch = torch.cat((in_patch.squeeze(0), out_patch.squeeze(0)), dim=0)
                else:
                    neighbor_patch = out_patch.squeeze(0)
                out_neighbors.append(neighbor_patch)

            # If fewer than k were found, repeat the best to keep a stable shape.
            if len(out_neighbors) == 0:
                fallback_channels = 2 if self.include_neighbor_inputs else 1
                fallback = torch.zeros((fallback_channels, q_depth, q_h, q_w), dtype=torch.float32)
                out_neighbors = [fallback for _ in range(k)]
            elif len(out_neighbors) < k:
                out_neighbors.extend([out_neighbors[-1]] * (k - len(out_neighbors)))

            # Each neighbor patch is [1, D, H, W] or [2, D, H, W]; stack along channels.
            neigh_tensor = torch.cat(out_neighbors[:k], dim=0)
            neighbors_batch.append(neigh_tensor)

        result = torch.stack(neighbors_batch, dim=0)
        return result.to(device=x.device, dtype=x.dtype)
        
        
    
def create_data_loader(
    image_dir,
    label_dir,
    channels,
    batch_size,
    device,
    num_slices=1,
    size=128,
    clear_cache=True,
    random_flip=False,
    random_shift=False
):
    # Use a transform-specific cache folder to avoid stale data reuse.
    cache_dir = ".monai_cache"
    if clear_cache and os.path.isdir(cache_dir):
        print(f"Clearing MONAI cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    case_ids = sorted([file[:9] for file in os.listdir(image_dir) if file.endswith(".nii.gz")])
    
    
    data_dicts = [
        {"input": os.path.join(image_dir, f"{case_id}_0000.nii.gz"),
         "output": os.path.join(label_dir, f"{case_id}_0001.nii.gz"),
         "case_id": case_id}
        for case_id in case_ids
    ]

    
    volume_transforms = _build_volume_transforms(size=size)
    
    # Create persistent part    
    print("Creating PersistentDataset...")
    persistent_ds = PersistentDataset(
        data=data_dicts,
        transform=volume_transforms,
        cache_dir=cache_dir
    )

    # Online slicing part
    sliced_ds = RandomSliceDataset(
        base_dataset=persistent_ds,
        channels=channels,
        size = size,
        num_slices = num_slices,
        random_flip=random_flip,
        random_shift=random_shift
    )

    train_loader = DataLoader(
        sliced_ds,
        batch_size=batch_size,   
        shuffle=True,   
        num_workers=16,
        collate_fn=list_data_collate, 
        pin_memory=True
    )
    return train_loader



def plot_slices_batch(data_torch, pixdims, slices, thicknesses=None, axes=None, **kwargs):
    for data_i, pixdim_i in zip(data_torch, pixdims):
        plot_slices(data_i.squeeze(), pixdim_i.cpu().numpy(), slices, thicknesses, axes, **kwargs)
        
        
def plot_slices(data_torch, pixdim, slices, thicknesses=None, axes=None, **kwargs):
    from data import plot_slices
    plot_slices(data_torch.permute(1, 2, 0), pixdim, slices, thicknesses, axes, **kwargs)