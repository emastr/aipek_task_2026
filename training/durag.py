
import time
import torch
torch.manual_seed(0)

import matplotlib.pyplot as plt
from monai.networks.nets import UNet

from src.architectures import DURAG, NeighborLoader
from src.plots import validation_plots
from src.processing import CONSTANTS
from src.data import create_data_loader

if __name__ == "__main__": # Avoids multiprocessing issues 
    epochs = 1000
    start_epoch = 2000
    size = 128
    channels = 64 # Number of slices per chunk; must be divisible by 32 for SwinUNETR
    batch_size = 16 # Number of chunks per GPU batch
    neighbor_count = 3
    save_path = "training/durag/"

    train_loader = create_data_loader(
        image_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_INP}",
        label_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_OUT}",
        channels = channels, # Number of slices per chunk
        batch_size = batch_size, # Number of chunks per GPU batch
        num_slices = 1, # Number of slices per chunk
        device = "cuda",
        size = size,
        random_flip = True,
        random_shift = True,
    )

    neighbor_loader = NeighborLoader(
        image_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_INP}",
        label_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_OUT}",
        channels = channels, # Number of slices per chunk
        neighbors = neighbor_count, # Number of nearest-neighbor feature maps
        device = "cuda",
        size = size,
        coarse_size = 32,
        include_neighbor_inputs = True,
    )

    valid_loader = create_data_loader(
        image_dir = f"{CONSTANTS.NORM_DATA_PATH_VAL_INP}",
        label_dir = f"{CONSTANTS.NORM_DATA_PATH_VAL_OUT}",
        channels = channels, # Number of slices per chunk
        batch_size = 1, # Number of chunks per GPU batch
        device = "cuda",
        random_flip = False,
        random_shift = False
    )

    trainval_loader = create_data_loader(
        image_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_INP}",
        label_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_OUT}",
        channels = channels, # Number of slices per chunk
        batch_size = 1, # Number of chunks per GPU batch
        device = "cuda",
        random_flip = True,
        random_shift = True
    )

    net = DURAG.init_large_durag(neighbor_loader, dropout=0.3)

    if start_epoch > 0:
        net.load_state_dict(torch.load(f"{save_path}epoch_{start_epoch}.pt"), strict=False)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )


    def loss_fn(ynet, y):
        y_01 = (y + 1)/2
        false_pos_weight =10.0
        false_neg_weight = 1.0
        both = y_01 #(y_01 + torch.clamp(ynet_01, 0, 1)) / 2
        loss_weight = (both * false_pos_weight + (1-both) * false_neg_weight)/(false_pos_weight + false_neg_weight)
        return torch.mean(loss_weight * (ynet -  y)**2) / torch.mean(loss_weight * y_01 ** 2)

        
    losses = []
    for epoch in range(start_epoch, start_epoch + epochs):
        for b, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            x = batch["input"].to("cuda", dtype=torch.float32, non_blocking=True)
            y = batch["output"].to("cuda", dtype=torch.float32, non_blocking=True)
            pixdim = batch["pixdim"].squeeze()
            case_id_batch = batch.get("case_id", None)
            
            
            inp = (x, case_id_batch)
            ynet = net(inp)
            loss = loss_fn(ynet, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            
            if b == 0:
                if epoch % 5 == 0:
                    # Save validation and training plots every 5 epochs
                    with torch.no_grad():
                        val_loss = validation_plots(net, valid_loader, f"{save_path}valid", epoch, loss_fn, neighbor_count)
                        train_loss = validation_plots(net, trainval_loader, f"{save_path}train", epoch, loss_fn, neighbor_count)
                        losses.append((epoch, val_loss, train_loss))
                        
                        plt.plot([e for e, _, _ in losses], [v for _, v, _ in losses], label="Validation Loss")
                        plt.plot([e for e, _, _ in losses], [t for _, _, t in losses], label="Train Loss")
                        plt.legend()
                        plt.savefig(f"{save_path}losses.png")
                        plt.close()
                    
                if epoch % 50 == 0:
                    # Save model checkpoint
                    torch.save(net.state_dict(), f"{save_path}epoch_{epoch}.pt")
                
            
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

        scheduler.step()
            