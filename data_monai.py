import torch
from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    Lambdad,
    Transposed,
    MapTransform,
    RandSpatialCropSamplesd,
    Resized
)
import os
import shutil


def _make_contiguous(x):
    return x.contiguous()

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
    def __init__(self, base_dataset, channels, size, num_slices=1):
        self.base_dataset = base_dataset
        self.patch_transform = Compose([
            VerticalPositionEmbedding(keys=["input"]),
            RandSpatialCropSamplesd(
                keys=["input", "output"],
                roi_size=(channels, size, size), 
                num_samples=num_slices,         
                random_size=False
            )
    ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        volume = self.base_dataset[idx]
        return self.patch_transform(volume)


def create_data_loader(
    image_dir,
    label_dir,
    channels,
    batch_size,
    device,
    num_slices=1,
    size=128,
    clear_cache=True
):
    # Use a transform-specific cache folder to avoid stale data reuse.
    cache_dir = ".monai_cache"
    if clear_cache and os.path.isdir(cache_dir):
        print(f"Clearing MONAI cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    case_ids = [file[:9] for file in os.listdir(image_dir) if file.endswith(".nii.gz")]
    
    
    data_dicts = [
        {"input": os.path.join(image_dir, f"{case_id}_0000.nii.gz"),
         "output": os.path.join(label_dir, f"{case_id}_0001.nii.gz")}
        for case_id in case_ids
    ]

    
    volume_transforms = Compose([
        LoadImaged(keys=["input", "output"]),
        EnsureChannelFirstd(keys=["input", "output"]),
        Transposed(keys=["input", "output"], indices=[0, 3, 1, 2]),
        Lambdad(keys=["input", "output"], func=_make_contiguous),
        CenterSpatialCropd(keys=["input", "output"], roi_size=(-1, 512, 512)),
        Resized(
            keys=["input", "output"],
            spatial_size=(-1, size, size),
            mode=("trilinear", "nearest"),
        ),
        ExtractSpacingd(keys=["input"], init_sizes=512, out_sizes=size),
    ])
    
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
        num_slices = num_slices
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