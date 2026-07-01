from typing import List, Sequence, Tuple, Optional
import torch
from torch import nn
import os
from src.data import NiiPoint
import torch.nn.functional as F
from monai.networks.nets import UNet
from src.data import _build_volume_transforms
import hashlib


class NiiKNN(nn.Module):
    
    def __init__(self, k, img_path, label_path, max_num=None):
        super().__init__()
        self.k = k
        self.label_path = label_path
        self.img_path = img_path
        self.max_num = max_num
        
    def forward(self, x: NiiPoint):
        files = os.listdir(self.img_path)
        # read dataset.json to get size
        num_data = len(files)
        dist = torch.zeros(num_data, device="cuda")
        for i,file in enumerate(files):
            print(f"Calculating distance for {file}...", end="\r")
            img = NiiPoint.from_path(os.path.join(self.img_path, file), "cuda")
            diff = x - img
            dist[i] = torch.mean(diff.data**2)**0.5
            if self.max_num is not None and i >= self.max_num:
                break
            
        sorted_idx = torch.argsort(dist)
        for i in range(self.k):
            idx = sorted_idx[i]
            file = files[idx]
            label_file = f"{file[:9]}_0001.nii.gz"
            xi = NiiPoint.from_path(os.path.join(self.img_path, file), "cuda")
            yi = NiiPoint.from_path(os.path.join(self.label_path, label_file), "cuda")    
            if i == 0:
                x = xi * (1 / self.k)
                y = yi * (1 / self.k)
            else:
                x = x + xi * (1 / self.k)
                y = y + yi * (1 / self.k)
        return x, y
    

class NiiCKNN(nn.Module):
    
    def __init__(self, k, kernel_size, img_path, label_path, max_num=None, max_batch=None):
        super().__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.label_path = label_path
        self.img_path = img_path
        self.max_num = max_num
        self.max_batch = max_batch
        self.case_ids = [file[:9] for file in os.listdir(self.img_path)]
        self.data_size = len(self.case_ids)
    
    def sliding_l2(self, x_pt: torch.Tensor, pixdims: tuple):
        kernel_size = [2 * int(self.kernel_size[i] / pixdims[i]) + 1 for i in range(3)]
        pad = [int((kernel_size[i] - 1) / 2) for i in range(3)]
        padding = (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0])
        x_pt = x_pt ** 2
        x_pt = F.pad(x_pt.unsqueeze(1), padding, mode='constant', value=0.0)
        weight = torch.ones((1, 1,) + tuple(kernel_size), dtype=x_pt.dtype, device=x_pt.device)
        sliding_sum = F.conv3d(x_pt, weight, stride=1)
        return sliding_sum.squeeze(1)

    
    def forward(self, x: NiiPoint):
        with torch.no_grad():
            files = os.listdir(self.img_path)
            # read dataset.json to get size
            sqdist = torch.full(x.data.shape, torch.inf, device="cuda")                     # track closest block distance
            x_est = torch.zeros_like(x.data, device="cuda")                                   # track closest block image
            y_est = torch.zeros_like(x.data, device="cuda")  
            
            
            for b in range(min(self.max_batch, len(files) // self.max_num)):
                img = torch.zeros((self.max_num, ) + x.data.shape, device="cuda")
                lab = torch.zeros((self.max_num, ) + x.data.shape, device="cuda")
                
                print(f"Processing batch {b + 1}/{min(self.max_batch, len(files) // self.max_num)}...", end="\r")
                for i in range(self.max_num):
                    xi, yi = self.load_case(i + b * self.max_num)
                    img[i, :, :, :] = xi.interpolate_to(x.data.shape).data
                    lab[i, :, :, :] = yi.interpolate_to(x.data.shape).data
                
                
                sqdists_all = self.sliding_l2(
                    x.data[None, :, :, :] - img, 
                    pixdims = x.header.get_zooms()
                    )
                
                sqdist_new, args = torch.min(sqdists_all, dim=0)
                x_new = torch.gather(img, 0, args[None, :, :, :]).squeeze(0)
                y_new = torch.gather(lab, 0, args[None, :, :, :]).squeeze(0)
                
                x_est = torch.where(sqdist_new < sqdist, x_new, x_est)
                y_est = torch.where(sqdist_new < sqdist, y_new, y_est)
                sqdist = torch.min(sqdist, sqdist_new)
                if self.max_num is not None and i >= self.max_num:
                    break
        
            return NiiPoint(x_est, x.affine, x.header), NiiPoint(y_est, x.affine, x.header)

    def load_case(self, idx):
        assert idx < self.data_size, f"Index {idx} out of range for dataset of size {self.data_size}"
        case_id = self.case_ids[idx]
        img_file = f"{case_id}_0000.nii.gz"
        label_file = f"{case_id}_0001.nii.gz"
        img = NiiPoint.from_path(os.path.join(self.img_path, img_file), "cuda")
        label = NiiPoint.from_path(os.path.join(self.label_path, label_file), "cuda")    
        return img, label
    
    
class DURAG(nn.Module):
    def __init__(self, net, neighbor_loader):
        super().__init__()
        self.net = net
        self.neighbor_loader = neighbor_loader

    def forward(self, inp):
        x, case_id_batch = inp
        x_nei = self.neighbor_loader.get_neighbors(x, ignore_case_id=case_id_batch)
        x_aug = torch.cat((x, x_nei), dim=1)
        return self.net(x_aug)
    
    @staticmethod
    def init_large_durag(neighbor_loader, dropout=0.1):
        neighbor_count = neighbor_loader.neighbors
        net = UNet(
            spatial_dims = 3, 
            in_channels = 2 + 2 * neighbor_count,
            out_channels = 1, 
            strides = (2, 2, 2),          # Downsampling factors
            channels = (64, 64, 128, 256), # old: (16, 32, 64, 128)  # Res: (128, 64, 32, 16)
            kernel_size=3, 
            up_kernel_size=3, 
            num_res_units=2,
            dropout=dropout
        ).to("cuda", dtype=torch.float32)
        return DURAG(net, neighbor_loader)


    def validate(self, valid_loader, criterion):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b, (x, y) in enumerate(valid_loader):
                x = x.to("cuda", dtype=torch.float32)
                y = y.to("cuda", dtype=torch.float32)
                inp = (x, None)
                y_hat = self(inp)
                loss = criterion(y_hat, y)
                val_loss += loss.item()
        self.train()
        return val_loss / len(valid_loader)


def pick_optimal_durag(durag, valid_loader, criterion, save_path):
    saved_models = [f for f in os.listdir(save_path) if f.endswith(".pt")]
    best_model = None
    best_model_file = None
    best_val_loss = float("inf")
    for model_file in saved_models:
        model_path = os.path.join(save_path, model_file)
        durag.net.load_state_dict(torch.load(model_path), strict=False)
        val_loss = durag.validate(valid_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model_file
            best_model_file = model_file
    return best_model, best_val_loss, best_model_file


class NeighborLoader:

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        channels: int, # Number of slices per chunk
        neighbors: int = 2, # Number of neighboring slices to include on each side
        device: str = "cuda",
        size: int = 128,
        coarse_size: int = 32,
        include_neighbor_inputs: bool = False,
        cache_dir: str = ".neighbor_cache",
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.channels = channels
        self.neighbors = neighbors
        self.device = device
        self.size = size
        self.coarse_size = coarse_size
        self.include_neighbor_inputs = include_neighbor_inputs
        self.cache_dir = cache_dir

        self._volume_transform = _build_volume_transforms(size=self.size)
        self._full_inputs: List[torch.Tensor] = []
        self._coarse_inputs: List[torch.Tensor] = []
        self._full_outputs: List[torch.Tensor] = []
        self._case_ids: List[str] = []
        self._case_id_to_index = {}
        self._neighbor_cache = {}
        self._neighbor_cache_path = None
        self._build_index()
        self._load_or_build_neighbor_cache()

    @staticmethod
    def _center_crop_or_pad_depth(volume: torch.Tensor, target_depth: int) -> torch.Tensor:
        depth = volume.shape[1]
        if depth == target_depth:
            return volume.contiguous()

        if depth > target_depth:
            start = (depth - target_depth) // 2
            return volume[:, start:start + target_depth].contiguous()

        pad_before = (target_depth - depth) // 2
        pad_after = target_depth - depth - pad_before
        return torch.nn.functional.pad(volume, (0, 0, 0, 0, pad_before, pad_after), value=-1.0).contiguous()

    def _cache_signature(self) -> str:
        payload = {
            "image_dir": os.path.abspath(self.image_dir),
            "label_dir": os.path.abspath(self.label_dir),
            "channels": self.channels,
            "neighbors": self.neighbors,
            "size": self.size,
            "coarse_size": self.coarse_size,
            "include_neighbor_inputs": self.include_neighbor_inputs,
            "case_ids": self._case_ids,
        }
        encoded = repr(payload).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:16]

    def _load_or_build_neighbor_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self._neighbor_cache_path = os.path.join(
            self.cache_dir,
            f"neighbor_cache_{self._cache_signature()}.pt",
        )

        if os.path.isfile(self._neighbor_cache_path):
            cache = torch.load(self._neighbor_cache_path, map_location="cpu")
            if cache.get("case_ids") == self._case_ids:
                self._neighbor_cache = cache.get("neighbors", {})
                print(f"Loaded neighbor cache: {self._neighbor_cache_path}")
                return

        print(f"Building neighbor cache: {self._neighbor_cache_path}")
        neighbor_cache = {}
        for case_id, case_idx in self._case_id_to_index.items():
            best = self._best_candidates(
                query_coarse=self._coarse_inputs[case_idx],
                k=self.neighbors,
                ignore_case_idx=case_idx,
            )
            neighbor_cache[case_id] = [self._case_ids[neighbor_case_idx] for _, neighbor_case_idx, _ in best]

        self._neighbor_cache = neighbor_cache
        torch.save({"case_ids": self._case_ids, "neighbors": self._neighbor_cache}, self._neighbor_cache_path)
        print(f"Saved neighbor cache: {self._neighbor_cache_path}")

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
            inp = self._center_crop_or_pad_depth(inp, self.channels)
            out = self._center_crop_or_pad_depth(out, self.channels)

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

            dist = torch.mean((coarse_volume - query_coarse) ** 2).item()
            best_per_case.append((dist, case_idx, 0))

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

        batch_size, _, _, q_h, q_w = x.shape
        k = self.neighbors if k is None else int(k)
        if k <= 0:
            raise ValueError("k must be > 0")

        ignore_case_ids = self._normalize_ignore_case_id(ignore_case_id, batch_size)

        query_for_match = x[:, :1].detach().to(device="cpu", dtype=torch.float32)
        query_for_match = torch.stack([
            self._center_crop_or_pad_depth(sample, self.channels)
            for sample in query_for_match
        ], dim=0)
        q_depth = int(query_for_match.shape[2])
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

            case_id = ignore_case_ids[b]
            cached_case_ids = self._neighbor_cache.get(case_id, None) if case_id is not None else None
            if cached_case_ids is not None:
                best = [
                    (0.0, self._case_id_to_index[neighbor_case_id], 0)
                    for neighbor_case_id in cached_case_ids
                    if neighbor_case_id in self._case_id_to_index and neighbor_case_id != case_id
                ]
                if len(best) < k:
                    search_best = self._best_candidates(
                        query_coarse=query_coarse[b],
                        k=k,
                        ignore_case_idx=ignore_case_idx,
                    )
                    seen_case_ids = {self._case_ids[case_idx] for _, case_idx, _ in best}
                    for dist, case_idx, start in search_best:
                        neighbor_case_id = self._case_ids[case_idx]
                        if neighbor_case_id in seen_case_ids or neighbor_case_id == case_id:
                            continue
                        best.append((dist, case_idx, start))
                        seen_case_ids.add(neighbor_case_id)
                        if len(best) == k:
                            break
                best = best[:k]
            else:
                best = self._best_candidates(
                    query_coarse=query_coarse[b],
                    k=k,
                    ignore_case_idx=ignore_case_idx,
                )

            out_neighbors: List[torch.Tensor] = []
            for _, case_idx, start in best:
                in_vol = self._full_inputs[case_idx]
                out_vol = self._full_outputs[case_idx]
                in_patch = in_vol.unsqueeze(0)
                out_patch = out_vol.unsqueeze(0)

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