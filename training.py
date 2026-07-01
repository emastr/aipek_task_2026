import os
import torch
from monai.networks.nets import UNet


def get_architecture(neighbor_count, dropout=0.1):
    net = UNet(
    spatial_dims = 3, 
    in_channels = 2 + 2 * neighbor_count,
    out_channels = 1, 
    strides = (2, 2, 2),          # Downsampling factors
    channels = (64, 64, 128, 256), # old: (16, 32, 64, 128)  # Res: (128, 64, 32, 16)
    kernel_size=3, 
    up_kernel_size=3, 
    num_res_units=2,
    dropout=dropout
    ).to("cuda", dtype=torch.float32)
    return net



def validate_model(net, valid_loader, neighbor_loader, criterion):
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for b, (x, y, case_id_batch) in enumerate(valid_loader):
            x = x.to("cuda", dtype=torch.float32)
            y = y.to("cuda", dtype=torch.float32)
            x_nei = neighbor_loader.get_neighbors(x, ignore_case_id=None)
            x_aug = torch.cat((x, x_nei), dim=1)
            y_hat = net(x_aug)
            loss = criterion(y_hat, y)
            val_loss += loss.item()
    val_loss /= len(valid_loader)
    return val_loss



def pick_optimal_model(net, valid_loader, neighbor_loader, criterion, save_path):
    saved_models = [f for f in os.listdir(save_path) if f.endswith(".pt")]
    best_model = None
    best_model_file = None
    best_val_loss = float("inf")
    for model_file in saved_models:
        model_path = os.path.join(save_path, model_file)
        net.load_state_dict(torch.load(model_path), strict=False)
        val_loss = validate_model(net, valid_loader, neighbor_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model_file
            best_model_file = model_file
    return best_model, best_val_loss, best_model_file