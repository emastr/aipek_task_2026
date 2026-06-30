"""Transform the predictions from the normalized data back to the original CTA data space using the transforms from data.py. This is necessary because the predictions are made on normalized data, and we want to visualize them in the original CTA space."""


from data import NiiPoint, DataTfmLibrary


def transform_vessel_to_cta(prediction_data_path, ncct_data_path, output_data_path):
    # Load the predictions and NCCT data
    vessel_normalized = NiiPoint.from_path(prediction_data_path, device="cpu")
    ncct_data = NiiPoint.from_path(ncct_data_path, device="cpu")
    vessel = DataTfmLibrary.inv_vessel_normalization_tfm(vessel_normalized)
    cta_data = DataTfmLibrary.inv_vessel_tfm(ncct_data, vessel)
    cta_data.save_to_path(output_data_path)


def transform_all(root_pred, root_ncct, root_output, id_subset=None):
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
            transform_vessel_to_cta(pred_path, ncct_path, output_path)

if __name__ == "__main__":
    
    root_pred = "nnUNet_raw/Predictions_val"
    root_ncct = "nnUNet_raw/Dataset/imagesTr"
    root_out = "nnUNet_raw/Predictions_val_small"
    models = ["cknn", "knn"]
    
    for model in models:
        root_pred_model = f"{root_pred}/images_{model}/"
        root_out_model = f"{root_out}/images_{model}/"
        
        transform_all(root_pred_model, root_ncct, root_out_model, [7, 10, 12])
