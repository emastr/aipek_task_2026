from __future__ import annotations

import math
import re
from pathlib import Path

import nibabel as nib
import numpy as np


def evaluate_metrics_to_markdown(
    model_name: str,
	vessel_norm_path: str | Path,
	ncct_norm_path: str | Path,
	ncct_path: str | Path,
	cta_path: str | Path,
	vessel_norm_pred_path: str | Path,
	cta_pred_path: str | Path,
	beta: float = 0.5,
) -> str:
	"""Evaluate TopCoW-style benchmark metrics and return a Markdown table.

	Metrics (as defined in README):
	  1) RMSE on transformed vessel images (y_tilde vs y_tilde_pred)
	  2) RMSE on CTA images (y vs y_pred)
	  3) IoU on transformed vessel masks thresholded at beta

	The function reports each metric as mean +- 2*std/sqrt(N), where N is the
	number of matched cases.
	"""
	metrics = evaluate_metrics_summary(
		vessel_norm_path=vessel_norm_path,
		ncct_norm_path=ncct_norm_path,
		ncct_path=ncct_path,
		cta_path=cta_path,
		vessel_norm_pred_path=vessel_norm_pred_path,
		cta_pred_path=cta_pred_path,
		beta=beta,
	)
	return f"| {model_name} | {metrics['rmse_tilde']} | {metrics['rmse_cta']} | {metrics['iou_tilde']} |"


def evaluate_metrics_summary(
	vessel_norm_path: str | Path,
	ncct_norm_path: str | Path,
	ncct_path: str | Path,
	cta_path: str | Path,
	vessel_norm_pred_path: str | Path,
	cta_pred_path: str | Path,
	beta: float = 0.5,
) -> dict[str, str]:
	# Keep all dataset paths explicit in the signature. ncct_norm_path and
	# ncct_path are accepted for compatibility with the experiment interface.
	_ = Path(ncct_norm_path)
	_ = Path(ncct_path)

	vessel_true = _index_cases(vessel_norm_path)
	vessel_pred = _index_cases(vessel_norm_pred_path)
	cta_true = _index_cases(cta_path)
	cta_pred = _index_cases(cta_pred_path)

	common_cases = sorted(set(vessel_true) & set(vessel_pred) & set(cta_true) & set(cta_pred))
	if not common_cases:
		raise ValueError("No overlapping case IDs found between target and prediction folders.")

	rmse_tilde_vals: list[float] = []
	rmse_cta_vals: list[float] = []
	iou_tilde_vals: list[float] = []

	for case_id in common_cases:
		y_tilde = _load_volume(vessel_true[case_id])
		y_tilde_pred = _load_volume(vessel_pred[case_id])
		y = _load_volume(cta_true[case_id])
		y_pred = _load_volume(cta_pred[case_id])

		if y_tilde.shape != y_tilde_pred.shape:
			raise ValueError(f"Shape mismatch for tilde case {case_id}: {y_tilde.shape} vs {y_tilde_pred.shape}")
		if y.shape != y_pred.shape:
			raise ValueError(f"Shape mismatch for CTA case {case_id}: {y.shape} vs {y_pred.shape}")

		rmse_tilde_vals.append(_relative_l1(y_tilde, y_tilde_pred) * 100)
		rmse_cta_vals.append(_relative_l1(y, y_pred) * 100)
		iou_tilde_vals.append(_iou_threshold(y_tilde, y_tilde_pred, beta=beta) * 100)

	return {
		"rmse_tilde": _format_mean_ci(rmse_tilde_vals * 1),
		"rmse_cta": _format_mean_ci(rmse_cta_vals * 1),
		"iou_tilde": _format_mean_ci(iou_tilde_vals * 1),
	}


def _index_cases(root: str | Path) -> dict[str, Path]:
	root_path = Path(root).expanduser().resolve()
	if not root_path.exists():
		raise FileNotFoundError(f"Path does not exist: {root_path}")

	case_map: dict[str, Path] = {}
	for f in sorted(root_path.rglob("*.nii.gz")):
		case_id = _extract_case_id(f.name)
		if case_id is not None and case_id not in case_map:
			case_map[case_id] = f
	return case_map


def _extract_case_id(name: str) -> str | None:
	m = re.search(r"case_(\d+)", name)
	return m.group(1) if m else None


def _load_volume(path: Path) -> np.ndarray:
	return nib.load(str(path)).get_fdata().astype(np.float32, copy=False)


def _relative_rmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
	num = np.linalg.norm(y_true - y_pred)
	den = max(np.linalg.norm(y_true), eps)
	return float(num / den)

def _relative_l1(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    #print(y_true.max(), y_pred.max())
    #print(y_true.min(), y_pred.min())
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.min(y_true)))
    return float(num / den)


def _iou_threshold(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
	a = (y_true + 1)/2 > beta
	b = (y_pred + 1)/2 > beta
	union = np.logical_or(a, b).sum()
	if union == 0:
		return 1.0
	inter = np.logical_and(a, b).sum()
	return float(inter / union)


def _format_mean_ci(values: list[float]) -> str:
	arr = np.asarray(values, dtype=np.float64)
	mean = float(arr.mean())
	std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
	ci = 2.0 * std / math.sqrt(max(arr.size, 1))
	return f"${mean:.1f}\\% \\pm {ci:.1f}\\%$"



        

