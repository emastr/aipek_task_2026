import nibabel as nib
import torch
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
    Resized,
)
import os
import shutil
from src.plots import plot_slices


_CLEARED_CACHE_DIRS = set()


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
    
class DeterministicSliceDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 base_dataset, 
                 channels, 
                 size, 
                 num_slices=1,
                random_flip=False,
                random_shift=False):
        self.base_dataset = base_dataset
        self.channels = channels
        self.size = size
        self.num_slices = num_slices
        self.vertical_position_embedding = VerticalPositionEmbedding(keys=["input"])

    @staticmethod
    def _center_crop_or_pad_depth(volume, target_depth):
        depth = volume.shape[1]
        if depth == target_depth:
            return volume.contiguous()

        if depth > target_depth:
            start = (depth - target_depth) // 2
            return volume[:, start:start + target_depth].contiguous()

        pad_before = (target_depth - depth) // 2
        pad_after = target_depth - depth - pad_before
        return torch.nn.functional.pad(volume, (0, 0, 0, 0, pad_before, pad_after), value=-1.).contiguous()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        volume = self.base_dataset[idx]
        sample = dict(volume)
        sample["input"] = self._center_crop_or_pad_depth(sample["input"], self.channels)
        sample["output"] = self._center_crop_or_pad_depth(sample["output"], self.channels)
        sample = self.vertical_position_embedding(sample)

        if "case_id" not in sample and "case_id" in volume:
            sample["case_id"] = volume["case_id"]
        sample["case_idx"] = torch.tensor(idx, dtype=torch.long)

        return [sample]


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
    cache_dir_abs = os.path.abspath(cache_dir)
    if clear_cache and cache_dir_abs not in _CLEARED_CACHE_DIRS and os.path.isdir(cache_dir):
        print(f"Clearing MONAI cache: {cache_dir}")
        shutil.rmtree(cache_dir)
        _CLEARED_CACHE_DIRS.add(cache_dir_abs)
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
    sliced_ds = DeterministicSliceDataset(
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
    from src.plots import plot_slices
    if torch.is_tensor(pixdim):
        pixdim = pixdim.detach().cpu().flatten().tolist()
    else:
        pixdim = list(pixdim)
    pixdim = tuple(float(v) for v in pixdim[:3])
    plot_slices(data_torch, pixdim, slices, thicknesses, axes, **kwargs)


class NiiPoint():
    @ staticmethod
    def from_path(path, device, dtype=torch.float32, tfm=None, ensure_lps=False):
        nii = nib.load(path)
        if ensure_lps:
            src_ornt = nib.orientations.io_orientation(nii.affine)
            lps_ornt = nib.orientations.axcodes2ornt(("L", "P", "S"))
            to_lps = nib.orientations.ornt_transform(src_ornt, lps_ornt)
            nii = nii.as_reoriented(to_lps)
        data_np = nii.get_fdata()
        data = torch.from_numpy(data_np).to(device, dtype=dtype)
        return NiiPoint(data, nii.affine, nii.header, device, dtype, tfm)

    def save_to_path(self, path):
        data_np = self.data.cpu().numpy()
        nii = nib.Nifti1Image(data_np, self.affine, self.header)
        nib.save(nii, path)

    def __init__(self, data, affine, header, device=None, dtype=torch.float32, tfm=None):
        if device is not None:
            self.data = data.to(device, dtype=dtype)
        else:
            self.data = data.to(dtype=dtype)
        if tfm is not None:
            self.data = tfm(self.data)
        self.affine = affine
        self.header = header

    def __add__(self, other):
        result = NiiPoint(self.data.clone(), self.affine, self.header, self.data.device)
        if result.data.shape != other.data.shape:
            other = other.interpolate_to(result.data.shape)
        result.data = self.data + other.data
        return result

    def __sub__(self, other):
        result = NiiPoint(self.data.clone(), self.affine, self.header, self.data.device)
        if result.data.shape != other.data.shape:
            other = other.interpolate_to(result.data.shape)
        result.data = self.data - other.data
        return result

    def __mul__(self, other):
        if other.__class__ == float or other.__class__ == int:
            result = NiiPoint(self.data.clone(), self.affine, self.header, self.data.device)
            result.data = self.data * other
            return result
        else:
            assert self.data.shape == other.data.shape, "Data shapes must match for multiplication"
            result = NiiPoint(self.data.clone(), self.affine, self.header, self.data.device)
            result.data = self.data * other.data
            return result


    def interpolate_to(self, target_shape):
        result = NiiPoint(self.data.clone(), self.affine, self.header, self.data.device)
        result.data = torch.nn.functional.interpolate(
            self.data.unsqueeze(0).unsqueeze(0),
            size=target_shape, mode='trilinear', align_corners=False
        ).squeeze(0).squeeze(0)
        return result

    def apply_tfm(self, tfm):
        new = NiiPoint(self.data.clone(), self.affine, self.header, self.data.device)
        new.data = tfm(new.data, new.affine, new.header)
        return new


    def plot_slices_nii(self, slices, thicknesses=None, axes=None,**kwargs):
        # Use affine-derived voxel sizes to keep spacing consistent after any reorientation.
        pixdim = tuple(float(v) for v in nib.affines.voxel_sizes(self.affine)[:3])
        return plot_slices(self.data, pixdim, slices, thicknesses, axes, **kwargs)