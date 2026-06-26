import torch
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class CONSTANTS:
    BACK = 0    # Background
    BONE = 100  # HU bone
    VESSEL_MAX = 400  # HU vessel max value for normalization
    NCCT_MIN = -30  # HU NCCT min value for normalization
    NCCT_MAX = 90   # HU NCCT max value for normalization
    NORM_DATA_PATH = "nnUNet_raw/Dataset_val/"
    NORM_DATA_PATH_VAL_INP = "nnUNet_raw/Dataset_val/imagesVl/"
    NORM_DATA_PATH_VAL_OUT = "nnUNet_raw/Dataset_val/labelsVl/"
    NORM_DATA_PATH_TRAIN_INP = "nnUNet_raw/Dataset_val/imagesTr/"
    NORM_DATA_PATH_TRAIN_OUT = "nnUNet_raw/Dataset_val/labelsTr/"
    
    
    

class NiiPoint():
    @ staticmethod
    def from_path(path, device, dtype=torch.float32, tfm=None):
        nii = nib.load(path)
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
        return plot_slices(self.data, self.header.get_zooms(), slices, thicknesses, axes, **kwargs)


def plot_slices(data_torch, pixdims, slices, thicknesses=None, axes=None, **kwargs):
    data_np = data_torch.cpu().numpy()
    idx = [int(slices[i] * data_np.shape[i]) for i in range(3)]

    # Axes: (X, Y, Z) = (width, height, depth)
    if thicknesses is None:
        slice_data = [
            data_np[idx[0], :, :],
            data_np[:, idx[1], :],
            data_np[:, :, idx[2]]
        ]
    elif thicknesses == "all":
        slice_data = [
            data_np.max(axis=0),
            data_np.max(axis=1),
            data_np.max(axis=2)
        ]
        
    else:
        slice_data = [
            data_np[idx[0]-thicknesses[0]:idx[0]+thicknesses[0], :, :].max(axis=0),
            data_np[:, idx[1]-thicknesses[1]:idx[1]+thicknesses[1], :].max(axis=1),
            data_np[:, :, idx[2]-thicknesses[2]:idx[2]+thicknesses[2]].max(axis=2)
        ]
    
    pos_slices = [idx[i] * pixdims[i] for i in range(3)]
    slice_widths = [pixdims[i] * data_np.shape[i] for i in range(3)]
    
    
    
    if axes is None:
        #axes = [
        #    plt.subplot(1, 3, i+1) for i in range(3)
        #]
        inv_wd = [1/w for w in slice_widths]
        sum_inv_wd = sum(inv_wd)
        ratios = [iw / sum_inv_wd for iw in inv_wd]
        max_ratio = max(ratios)
        
        plt.figure(figsize=(10, 10*max_ratio))
        gs = gridspec.GridSpec(1, 3, width_ratios=ratios)
        axes = [plt.gcf().add_subplot(gs[0,i]) for i in range(3)]
    for i in range(3):
        j2, j1 = [j for j in range(3) if j != i]
        axes[i].imshow(slice_data[i], extent=[0, slice_widths[j1], 0, slice_widths[j2]], **kwargs)
        axes[i].plot([pos_slices[j1], pos_slices[j1]], [0, slice_widths[j2]], 'steelblue', linewidth=1)
        axes[i].plot([0, slice_widths[j1]], [slice_widths[j2]-pos_slices[j2], slice_widths[j2]-pos_slices[j2]], 'steelblue', linewidth=1)
        axes[i].set_title(f'{["X", "Y", "Z"][i]} Slice')
        axes[i].set_aspect("equal")
        axes[i].axis('off')
    
    plt.tight_layout()
    return plt.gca()


class DataTfmLibrary: 
    @staticmethod
    def vessel_tfm(ncct_data, cta_data):
        clamp = (
            lambda data, affine, header:
                torch.clamp(data, min=0.0)
            )
        bone_mask = ncct_data.apply_tfm(
            BaseTfmLibrary.bone_mask_tfm
        )
        diff = (
            bone_mask * (
                cta_data.apply_tfm(clamp) -
                ncct_data.apply_tfm(clamp)
                ).apply_tfm(clamp)
        )
        return diff
    
    @staticmethod
    def inv_vessel_tfm(ncct_data, diff_data):
        return ncct_data + diff_data 
    
    
    @staticmethod
    def vessel_normalization_tfm(vessel_data):
        return vessel_data.apply_tfm(
            BaseTfmLibrary.affine(0.0, CONSTANTS.VESSEL_MAX)
        ).apply_tfm(
            BaseTfmLibrary.clip(-1.0, 1.0)
        )
        
    @staticmethod
    def inv_vessel_normalization_tfm(vessel_data):
        return vessel_data.apply_tfm(
            BaseTfmLibrary.inv_affine(0.0, CONSTANTS.VESSEL_MAX)
        )
        
    
    @staticmethod
    def ncct_normalization_tfm(ncct_data):
        return ncct_data.apply_tfm(
            BaseTfmLibrary.affine(CONSTANTS.NCCT_MIN, CONSTANTS.NCCT_MAX)
        ).apply_tfm(
            BaseTfmLibrary.clip(-1.0, 1.0)
        )
    
    @staticmethod
    def inv_ncct_normalization_tfm(ncct_data):
        return ncct_data.apply_tfm(
            BaseTfmLibrary.inv_affine(CONSTANTS.NCCT_MIN, CONSTANTS.NCCT_MAX)
        )
    
    

class BaseTfmLibrary:
    
    @staticmethod
    def expand_mask(mask, thickness, pixdim):
        # Expand the mask by a given thickness in all directions
        kernel_size = [2 *int(thickness / pixdim[i]) + 1 for i in range(3)]
        padding = [k // 2 for k in kernel_size]
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        expanded_mask = torch.nn.functional.max_pool3d(mask.float(), kernel_size=kernel_size, stride=1, padding=padding)
        return expanded_mask.squeeze(0).squeeze(0) > 0

    @staticmethod
    def anisotropic_gaussian_blur(input, thickness, pixdim):
        # Create a Gaussian kernel with anisotropic standard deviations
        sigma = [thickness / pixdim[i] for i in range(3)]
        kernel_size = [2*int(s) + 1 for s in sigma]
        grid = torch.meshgrid([torch.arange(-k//2 + 1, k//2 + 1) for k in kernel_size], indexing='ij')
        kernel = torch.exp(-0.5 * sum((g / s)**2 for g, s in zip(grid, sigma))).to(input.device)
        kernel /= kernel.sum()  # Normalize the kernel

        # Apply the Gaussian blur using convolution
        input = input.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        blurred = torch.nn.functional.conv3d(input.float(), kernel.unsqueeze(0).unsqueeze(0), padding=[k//2 for k in kernel_size])
        return blurred.squeeze(0).squeeze(0)

    @staticmethod
    def bone_mask_tfm(data, affine, header):
        pixdim = header.get_zooms()
        data = data > CONSTANTS.BONE
        data = BaseTfmLibrary.expand_mask(data, 1, pixdim).float()
        data = BaseTfmLibrary.anisotropic_gaussian_blur(data, 2, pixdim)
        return 1.- data.to(torch.float32)

    @staticmethod    
    def split(cutoff):
        def _split(data, affine=None, header=None):
            return (data > cutoff) * data
        return _split
    
    
    @staticmethod
    def clip(lower, upper):
        def _tfm(data, affine=None, header=None):
            return torch.clamp(data, min=lower, max=upper)
        return _tfm
    
    @staticmethod
    def affine(lower, upper):
        def _tfm(data, affine=None, header=None):
            mid = (upper + lower)/2
            haf = (upper - lower)/2
            return (data - mid)/haf
        return _tfm

    @staticmethod
    def inv_affine(lower, upper):
        def _tfm(data, affine=None, header=None):
            mid = (upper + lower)/2
            haf = (upper - lower)/2
            return data * haf + mid
        return _tfm