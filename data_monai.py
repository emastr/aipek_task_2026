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
    RandSpatialCropSamplesd
)
import os

class ExtractSpacingd(MapTransform):
    """
    Custom transform to pull the pixdim out of the NIfTI header
    and save it as a new key in the data dictionary.
    """
    def __init__(self, keys, meta_key_postfix="meta_dict"):
        super().__init__(keys)
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)
        img = d["input"]
        spacing = img.meta["pixdim"][1:4]
        d[f"pixdim"] = spacing
        return d
    
    
class RandomSliceDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, channels):
        self.base_dataset = base_dataset
        self.patch_transform = Compose([
            RandSpatialCropSamplesd(
                keys=["input", "output"],
                roi_size=(channels, 512, 512), 
                num_samples=1,         
                random_size=False
        )
    ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        volume = self.base_dataset[idx]
        return self.patch_transform(volume)


def create_data_loader(image_dir, label_dir, channels, batch_size, device):
    cache_dir = "../.monai_cache"  # Directory to store cached data
    case_ids = [file[:9] for file in os.listdir(image_dir) if file.endswith(".nii.gz")]
    
    
    data_dicts = [
        {"input": os.path.join(image_dir, f"{case_id}_0000.nii.gz"),
         "output": os.path.join(label_dir, f"{case_id}_0001.nii.gz")}
        for case_id in case_ids
    ]

    
    volume_transforms = Compose([
        LoadImaged(keys=["input", "output"]),
        EnsureChannelFirstd(keys=["input", "output"]),
        ExtractSpacingd(keys=["input"]),
        Transposed(keys=["input", "output"], indices=[0, 3, 1, 2]),
        Lambdad(keys=["input", "output"], func=lambda x: x.contiguous()),
        CenterSpatialCropd(keys=["input", "output"], roi_size=(-1, 512, 512)), 
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
        channels=channels
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