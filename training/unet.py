from matplotlib import image as im


if __name__ == "__main__":
    import sys
    sys.path.append('')
    sys.path.append('../')

    import torch
    torch.manual_seed(0)

    import matplotlib.pyplot as plt
    from data import CONSTANTS
    from monai.networks.nets import BasicUNet
    from data_monai import create_data_loader, plot_slices
    
    epochs = 2_000
    epoch_start = 695
    size = 128
    channels = 8 # Number of slices per chunk
    batch_size = 7 # Number of chunks per GPU batch
    num_slices = 20
    save_path = "training/unet/"
    train_loader = create_data_loader(
        image_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_INP}",
        label_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_OUT}",
        channels = channels, # Number of slices per chunk
        batch_size = batch_size, # Number of chunks per GPU batch
        num_slices = num_slices, # Number of slices per chunk
        device = "cuda",
        size = size
    )
    valid_loader = create_data_loader(
        image_dir = f"{CONSTANTS.NORM_DATA_PATH_VAL_INP}",
        label_dir = f"{CONSTANTS.NORM_DATA_PATH_VAL_OUT}",
        channels = channels, # Number of slices per chunk
        batch_size = 1, # Number of chunks per GPU batch
        device = "cuda"
    )

    net = BasicUNet(
        spatial_dims=2,
        in_channels= channels * 2,      # X and positional encoding
        out_channels= channels,             # Y0 - Y1
        features=(64, 64, 128, 256, 512, 64),#(32, 32, 64, 128, 256, 32),
    ).to("cuda", dtype=torch.float32)
    
    if epoch_start is not None:
        net.load_state_dict(torch.load(f"{save_path}epoch_{epoch_start}.pt", map_location="cuda"))
    
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-7,
    )


    def loss_fn(ynet, y):
        return torch.mean((ynet -  y)**2)


    epch_range = range(epoch_start, epochs) if epoch_start is not None else range(epochs)
        
    for epoch in epch_range:
        for b, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            x = batch["input"].squeeze().to("cuda", dtype=torch.float32, non_blocking=True).view(-1, channels*2, size, size)
            y = batch["output"].squeeze().to("cuda", dtype=torch.float32, non_blocking=True)
            pixdim = batch["pixdim"].squeeze()

            timesteps = torch.ones(x.shape[0]).to("cuda", dtype=torch.float32)

            ynet = net(x)
            loss = loss_fn(ynet, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0 and b == 0:
                
                with torch.no_grad():
                    iter_val = iter(valid_loader)
                    for i in range(3):
                        val_batch = next(iter_val)
                        x_val = val_batch["input"].squeeze(1).to("cuda", dtype=torch.float32, non_blocking=True).view(-1, channels*2, size, size)
                        y_val = val_batch["output"].squeeze(1).to("cuda", dtype=torch.float32, non_blocking=True)
                        pixdim = val_batch["pixdim"].squeeze()
                        ynet_val = net(x_val)
                        val_loss = loss_fn(ynet_val, y_val)
                        
                        title = f"Epoch {epoch} Prediction"
                        
                        slices = (0.4, 0.4, 0.4)
                        thicknes = (10, 10, 10)
                        kwargs = {"vmin": -1, "vmax": 1, "cmap": "gray"}
                        plot_slices(ynet_val.squeeze(), pixdim, slices, (1,1,1), **kwargs)
                        plt.gca().set_title(title)
                        plt.gcf().savefig(f"{save_path}pred{i}.png")
                        plot_slices(y_val.cpu().squeeze(), pixdim, slices, thicknes, **kwargs)
                        plt.gca().set_title(title)
                        plt.gcf().savefig(f"{save_path}out{i}.png")
                        plot_slices(x_val[:, :channels].cpu().squeeze(), pixdim, slices, thicknes, **kwargs)
                        plt.gca().set_title(title)
                        plt.gcf().savefig(f"{save_path}inp{i}.png")
                        plt.close('all')
                        
                    plt.figure(figsize=(15, 10))
                    for i in range(3):
                        for j, target in enumerate(["pred", "out", "inp"]):
                            plt.subplot(3, 3, i*3 + j + 1)
                            plt.imshow(im.imread(f"{save_path}{target}{i}.png"))
                            plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(f"{save_path}examples.png")
                    plt.close('all')

                torch.save(net.state_dict(), f"{save_path}epoch_{epoch}.pt")
                
            
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, LR: {current_lr:.6e}")

        scheduler.step()
            