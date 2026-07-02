import torch

from src.data import NiiPoint

DATA_ROOT = "nnUNet_raw"

class CONSTANTS:
    BACK = 0    # Background
    BONE = 100  # HU bone
    VESSEL_MAX = 400  # HU vessel max value for normalization
    NCCT_MIN = -30  # HU NCCT min value for normalization
    NCCT_MAX = 90   # HU NCCT max value for normalization
    DATA_PATH = f"{DATA_ROOT}/Dataset/"
    NORM_DATA_PATH = f"{DATA_ROOT}/Dataset_val/"
    NORM_DATA_PATH_VAL_INP = f"{DATA_ROOT}/Dataset_val/imagesVl/"
    NORM_DATA_PATH_VAL_OUT = f"{DATA_ROOT}/Dataset_val/labelsVl/"
    NORM_DATA_PATH_TRAIN_INP = f"{DATA_ROOT}/Dataset_val/imagesTr/"
    NORM_DATA_PATH_TRAIN_OUT = f"{DATA_ROOT}/Dataset_val/labelsTr/"
    NORM_DATA_PATH_TEST_INP = f"{DATA_ROOT}/Dataset_val/imagesTs/"
    NORM_DATA_PATH_TEST_OUT = f"{DATA_ROOT}/Dataset_val/labelsTs/"
    NORM_PRED_PATH = f"{DATA_ROOT}/Predictions_val/"
    TEST_PRED_PATH = f"{DATA_ROOT}/Predictions_test/"
    PRED_PATH = f"{DATA_ROOT}/Predictions_val_small/"
    

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


def transform_vessel_to_cta(prediction_data_path, ncct_data_path, output_data_path, to_int=False):
    # Load the predictions and NCCT data
    vessel_normalized = NiiPoint.from_path(prediction_data_path, device="cpu")
    vessel_normalized.apply_tfm(lambda x, h, a: torch.clamp(x, min=-1.0, max=1.0))
    ncct_data = NiiPoint.from_path(ncct_data_path, device="cpu")
    vessel = DataTfmLibrary.inv_vessel_normalization_tfm(vessel_normalized)
    cta_data = DataTfmLibrary.inv_vessel_tfm(ncct_data, vessel)
    if to_int:
        cta_data.data = cta_data.data.to(torch.int16)
    cta_data.save_to_path(output_data_path)


def transform_all(root_pred, root_ncct, root_output, id_subset=None, to_int=False):
    import os
    from pathlib import Path

    root_pred = Path(root_pred)
    root_ncct = Path(root_ncct)
    root_output = Path(root_output)

    for pred_file in os.listdir(root_pred):
        if pred_file.endswith(".nii.gz"):
            case_id = pred_file.split("_")[1]
            if id_subset is not None and int(case_id) not in id_subset:
                continue
            ncct_file = f"case_{case_id}_0000.nii.gz"
            output_file = f"case_{case_id}_0001.nii.gz"
            pred_path = root_pred / pred_file
            ncct_path = root_ncct / ncct_file
            output_path = root_output / output_file
            transform_vessel_to_cta(pred_path, ncct_path, output_path, to_int)