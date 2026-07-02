import os
import matplotlib.gridspec as gridspec
import matplotlib.image as im
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Any


WARNING = (
    """
    The plotting functions break depending on versions of torch and monai, and backend for nii loading.
    They are still useful, but become heavily distorted if the nii loading backend is not set to nibabel. 
    If you see distorted plots, try setting the environment variable:
    """
)

# print warning
print(WARNING)

def plot_from_path(path, slices, thicknesses=None, axes=None, **kwargs):
    from src.data import NiiPoint
    nii = NiiPoint.from_path(path, device="cpu", dtype=torch.float32, ensure_lps=True)
    return nii.plot_slices_nii(slices, thicknesses=thicknesses, axes=axes, **kwargs)


def plot_cta_and_segmentation(
    input_case_path: str | Path,
    prediction_case_path: str | Path,
    slices = None,
    thickness = None,
    device: str = "cpu",
    figsize: tuple[int, int] = (12, 12),
    **kwargs
) -> Any:
    """Plot CTA-only, segmentation-only, and overlay views for one case.

    The function uses plot utilities from data.py (NiiPoint.plot_slices_nii) and
    creates three rows of orthogonal slices:
      1) CTA only,
      2) segmentation only,
      3) overlay of segmentation on CTA.

    Args:
        input_case_path: Path to input image NIfTI (.nii.gz).
        prediction_case_path: Path to predicted mask NIfTI (.nii.gz).
        device: Torch device for loading (default "cpu").
        figsize: Figure size for the 3x3 panel.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import torch

    from src.data import NiiPoint

    input_case_path = Path(input_case_path).expanduser().resolve()
    prediction_case_path = Path(prediction_case_path).expanduser().resolve()

    if not input_case_path.exists():
        raise FileNotFoundError(f"Input case not found: {input_case_path}")
    if not prediction_case_path.exists():
        raise FileNotFoundError(f"Prediction case not found: {prediction_case_path}")

    img = NiiPoint.from_path(input_case_path, device=torch.device(device), dtype=torch.float32, ensure_lps=True)
    pred = NiiPoint.from_path(prediction_case_path, device=torch.device(device), dtype=torch.float32, ensure_lps=True)

    if img.data.shape != pred.data.shape:
        pred = pred.interpolate_to(img.data.shape)

    # Pick a representative location from the prediction mask if available.
    nz = torch.nonzero(pred.data > 0)
    if slices is not None:
        slices = [float(s) for s in slices]
    elif nz.numel() > 0:
        center = nz.float().mean(dim=0)
        slices = [
            float(center[0].item() / max(1, img.data.shape[0] - 1)),
            float(center[1].item() / max(1, img.data.shape[1] - 1)),
            float(center[2].item() / max(1, img.data.shape[2] - 1)),
        ]
    else:
        slices = [0.5, 0.5, 0.5]

    fig, axes = plt.subplots(1, 3, figsize=figsize)


    thickness = (1, 1, 1) if thickness is None else tuple(thickness)
    # Row 1: CTA only
    #img.plot_slices_nii(slices, thickness, axes=axes[0], cmap="gray")

    # Row 2: overlay
    img.plot_slices_nii(slices, thickness, axes=axes, **kwargs)
    data = pred.data.clone()
    data = torch.where(data==0, torch.nan, data)
    pred_overlay = NiiPoint(data, pred.affine, pred.header, pred.data.device)
    overlay_cmap = plt.get_cmap("Set3").copy()
    overlay_cmap.set_bad(alpha=0.0)
    pred_overlay.plot_slices_nii(slices,
                                 (data.shape[0]//10, data.shape[1]//10, data.shape[2]//10),
                                 axes=axes,
                                 cmap=overlay_cmap,
                                 alpha=1.0,
                                 interpolation="nearest")

    return fig


def plot_slices(data_torch, pixdims, slices, thicknesses=None, axes=None, **kwargs):
    data_np = data_torch.detach().cpu().numpy()
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
            np.nanmax(data_np[idx[0]-thicknesses[0]:idx[0]+thicknesses[0], :, :], axis=0),
            np.nanmax(data_np[:, idx[1]-thicknesses[1]:idx[1]+thicknesses[1], :], axis=1),
            np.nanmax(data_np[:, :, idx[2]-thicknesses[2]:idx[2]+thicknesses[2]], axis=2)
        ]

    pos_slices = [idx[i] * pixdims[i] for i in range(3)]
    slice_widths = [pixdims[i] * data_np.shape[i] for i in range(3)]
    plot_widths = [slice_widths[i] for i in [2, 2, 1]]


    if axes is None:
        #axes = [
        #    plt.subplot(1, 3, i+1) for i in range(3)
        #]
        sum_wd = sum(plot_widths)
        ratios = [iw / sum_wd for iw in plot_widths]
        max_ratio = max(ratios)

        plt.figure(figsize=(10, 10*max_ratio))
        gs = gridspec.GridSpec(1, 3, width_ratios=ratios)
        axes = [plt.gcf().add_subplot(gs[0,i]) for i in range(3)]
    for i in range(3):
        j2, j1 = [j for j in range(3) if j != i]
        axes[i].imshow(slice_data[i], extent=[0, slice_widths[j1], 0, slice_widths[j2]], **kwargs)
        axes[i].plot([pos_slices[j1], pos_slices[j1]], [0, slice_widths[j2]], 'red', linewidth=1)
        axes[i].plot([0, slice_widths[j1]], [slice_widths[j2]-pos_slices[j2], slice_widths[j2]-pos_slices[j2]], 'red', linewidth=1)
        axes[i].set_title(f'{["X", "Y", "Z"][i]} Slice')
        axes[i].set_aspect("equal")
        axes[i].axis('off')

    plt.tight_layout()
    return plt.gca()


def validation_plots(durag, loader, path, epoch, loss_fn, neighbor_count, num_vals=3):
    iter_val = iter(loader)
    loss_val = 0.0
    os.makedirs(f"{path}_img", exist_ok=True)
    for i in range(num_vals):
        val_batch = next(iter_val)
        x_val = val_batch["input"].to("cuda", dtype=torch.float32, non_blocking=True)
        y_val = val_batch["output"].to("cuda", dtype=torch.float32, non_blocking=True)
        pixdim = val_batch["pixdim"].squeeze()
        case_id_batch = val_batch.get("case_id", None)
        
        x_nei = durag.neighbor_loader.get_neighbors(x_val, ignore_case_id=case_id_batch)
        x_val_aug = torch.cat((x_val, x_nei), dim=1)
        ynet_val = durag.net(x_val_aug)
        val_loss = loss_fn(ynet_val, y_val)
        loss_val += val_loss.item()

        title = f"Epoch {epoch} Prediction"

        slices = (0.4, 0.4, 0.4)
        thicknes = (10, 10, 10)
        kwargs = {"vmin": -1, "vmax": 1, "cmap": "gray"}
        plot_slices(ynet_val[0, 0].cpu(), pixdim, slices, (1,1,1), **kwargs)
        plt.gca().set_title(title)
        plt.gcf().savefig(f"{path}_img/pred{i}.png")
        plot_slices(y_val[0, 0].cpu(), pixdim, slices, thicknes, **kwargs)
        plt.gca().set_title(title)
        plt.gcf().savefig(f"{path}_img/out{i}.png")
        plot_slices(x_val[0, 0].cpu(), pixdim, slices, thicknes, **kwargs)
        plt.gca().set_title(title)
        plt.gcf().savefig(f"{path}_img/inp{i}.png")
        for j in range(neighbor_count):
            plot_slices(x_nei[0, 2 * j].cpu(), pixdim, slices, thicknes, **kwargs)
            plt.gca().set_title(title)
            plt.gcf().savefig(f"{path}_img/nei{j}_inp{i}.png")
            plot_slices(x_nei[0, 2 * j + 1].cpu(), pixdim, slices, thicknes, **kwargs)
            plt.gca().set_title(title)
            plt.gcf().savefig(f"{path}_img/nei{j}_out{i}.png")
        plt.close('all')

    loss_val = loss_val / num_vals
    
    
    col_count = 3 + 2 * neighbor_count
    plt.figure(figsize=(10 * col_count, 10 * num_vals))
    for i in range(num_vals):
        neighbor_targets = []
        for k in range(neighbor_count):
            neighbor_targets.extend([f"nei{k}_inp", f"nei{k}_out"])
        for j, target in enumerate(["pred", "out", "inp"] + neighbor_targets):
            plt.subplot(3, col_count, i*col_count + j + 1)
            plt.imshow(im.imread(f"{path}_img/{target}{i}.png"))
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{path}_examples.png")
    plt.close('all')
    return loss_val