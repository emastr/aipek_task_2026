from src.segmentation import run_topcow_inference
from src.processing import CONSTANTS
import os

if __name__ == "__main__":

    cases = ["cta", "ncct", "cknn", "knn", "durag"]
    for case, filterc in zip(
        cases, 
        # Filter out the ones that we already ran
        [False, True, True, True, True]): 
        data_path = f"{CONSTANTS.NORM_PRED_PATH}images_{case}/"
        output_path = f"{CONSTANTS.NORM_PRED_PATH}segment_new_{case}/"
        os.makedirs(output_path, exist_ok=True)

        
        if filterc:
            print("===================================")
            print("===================================")
            print("===================================")
            print(f"Running inference for case: {case}")
            print("===================================")
            print("===================================")
            print("===================================")
            run_topcow_inference(
                data_path,
                output_path,
                track="ct",
                repo_root="",
                cases=[7, 10, 12]
            )
