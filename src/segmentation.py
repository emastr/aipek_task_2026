from __future__ import annotations

import os
import re
import subprocess
import glob
from pathlib import Path
from typing import Any
import nibabel as nib


def run_topcow_inference(
    data_path: str | Path,
    output_path: str | Path,
    track: str = "ct",
    repo_root: str | Path = "",
    cases: list[int] | None = None,
) -> list[Path]:
    """Stage input data into TopCoWSubmissions, run inference, and move outputs.

    Input images are reoriented to LPS+ before inference.
    Prediction masks are reoriented back to each case's original orientation
    before being written into output_path.

    Args:
        data_path: A .nii.gz file or a folder containing .nii.gz files.
        output_path: Folder where final prediction files will be moved.
        track: Either "ct" or "mr".
        repo_root: Path to the repository root containing TopCoWSubmissions.

    Returns:
        A list of moved prediction paths in output_path.
    """
    track = track.lower().strip()
    if track not in {"ct", "mr"}:
        raise ValueError("track must be 'ct' or 'mr'")

    repo_root = Path(repo_root).expanduser().resolve()
    topcow_root = repo_root / "TopCoWSubmissions"
    input_dir = topcow_root / "input" / ("head-ct-angio" if track == "ct" else "head-mr-angio")
    model_output_dir = topcow_root / "output" / "images" / "cow-multiclass-segmentation"
    output_dir = Path(output_path).expanduser().resolve()

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = _collect_nifti_files(data_path)
    if not source_files:
        raise FileNotFoundError(f"No .nii.gz files found in {data_path}")

    # Ensure this run only processes the files provided for this call.
    for p in input_dir.glob("*.nii.gz"):
        p.unlink()

    staged_names: list[str] = []
    orientation_meta: dict[str, dict[str, Any]] = {}
    for src in source_files:
        if cases is not None:
            match = re.search(r"case_(\d{4})_", src.name)
            if not match:
                raise ValueError(f"Filename {src.name} does not contain a valid case number.")
            case_number = int(match.group(1))
            if case_number not in cases:
                print(f"Skipping case {case_number} from file {src.name} as it's not in the specified cases list.")
                continue  # Skip this file if its case number is not in the specified list
            else:
                print(f"Processing case {case_number} from file {src.name}")
        staged_name = _to_topcow_channel_name(src.name)
        _save_as_lps(src, input_dir / staged_name)
        staged_names.append(staged_name)
        orientation_meta[staged_name] = {
            "original_name": src.name,
            "original_ornt": _get_orientation(src),
        }

    env = os.environ.copy()
    env["det_data"] = str(topcow_root / "nnDet" / "input")
    env["det_models"] = str(topcow_root / "nnDet" / "model")
    env["OMP_NUM_THREADS"] = "1"
    env["det_num_threads"] = "12"
    env["nnUNet_raw"] = str(topcow_root / "nnUNet" / "input" / "image")
    env["nnUNet_preprocessed"] = str(topcow_root / "nnUNet" / "input" / "preprocessed")
    env["nnUNet_results"] = str(topcow_root / "nnUNet" / "model")

    py_paths = [
        str(topcow_root / "nnDet" / "nnDetection"),
        str(topcow_root / "nnUNet" / ".segenv" / "lib" / "python3.11" / "site-packages"),
    ]
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(py_paths + ([existing_pp] if existing_pp else []))

    inference_py = topcow_root / "nnDet" / ".detenv" / "bin" / "python"
    subprocess.run([str(inference_py), "inference.py"], cwd=topcow_root, env=env, check=True)

    moved_paths: list[Path] = []
    for staged_name in staged_names:
        pred = model_output_dir / staged_name
        if pred.exists():
            original_name = str(orientation_meta[staged_name]["original_name"])
            original_ornt = orientation_meta[staged_name]["original_ornt"]
            dst = output_dir / original_name
            _save_from_lps_to_orientation(pred, dst, original_ornt)
            pred.unlink()
            moved_paths.append(dst)

    if not moved_paths:
        raise RuntimeError("Inference finished but no output files were moved.")

    return moved_paths


def _collect_nifti_files(data_path: str | Path) -> list[Path]:
    raw = os.path.expanduser(str(data_path))

    # Support shell-style patterns, e.g. /path/to/*.nii.gz
    if any(c in raw for c in "*?[]"):
        matches = [Path(m).resolve() for m in glob.glob(raw, recursive=True)]
        files = [m for m in matches if m.is_file() and m.name.endswith(".nii.gz")]
        return sorted(set(files))

    p = Path(raw).resolve()
    if p.is_file() and p.name.endswith(".nii.gz"):
        return [p]
    if p.is_dir():
        # Recursive search allows nested folder layouts.
        return sorted(f.resolve() for f in p.rglob("*.nii.gz") if f.is_file())
    return []


def _to_topcow_channel_name(filename: str) -> str:
    if filename.endswith("_0000.nii.gz"):
        return filename
    if re.search(r"_\d{4}\.nii\.gz$", filename):
        return re.sub(r"_\d{4}\.nii\.gz$", "_0000.nii.gz", filename)
    if filename.endswith(".nii.gz"):
        return filename.replace(".nii.gz", "_0000.nii.gz")
    return filename


def _get_orientation(path: Path) -> Any:
    img = nib.load(str(path))
    return nib.orientations.io_orientation(img.affine)


def _save_as_lps(src_path: Path, dst_path: Path) -> None:
    src_img = nib.load(str(src_path))
    orig_ornt = nib.orientations.io_orientation(src_img.affine)
    lps_ornt = nib.orientations.axcodes2ornt(("L", "P", "S"))
    to_lps = nib.orientations.ornt_transform(orig_ornt, lps_ornt)
    lps_img = src_img.as_reoriented(to_lps)
    nib.save(lps_img, str(dst_path))


def _save_from_lps_to_orientation(src_lps_path: Path, dst_path: Path, target_ornt: Any) -> None:
    src_img = nib.load(str(src_lps_path))
    lps_ornt = nib.orientations.axcodes2ornt(("L", "P", "S"))
    to_target = nib.orientations.ornt_transform(lps_ornt, target_ornt)
    target_img = src_img.as_reoriented(to_target)
    nib.save(target_img, str(dst_path))

