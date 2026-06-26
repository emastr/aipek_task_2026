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
        
    
    channels = 8 # Number of slices per chunk
    batch_size = 12 # Number of chunks per GPU batch
    train_loader = create_data_loader(
        image_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_INP}",
        label_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_OUT}",
        channels = channels, # Number of slices per chunk
        batch_size = batch_size, # Number of chunks per GPU batch
        device = "cuda"
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
        in_channels= channels,      # Yt, X and t.
        out_channels= channels,             # Y0 - Y1
        features=(32, 32, 64, 128, 256, 32),
    ).to("cuda", dtype=torch.float32)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4)


    def loss_fn(ynet, y):
        return torch.mean((ynet -  y)**2)


        
    for epoch in range(500):
        for b, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            x = batch["input"].squeeze().to("cuda", dtype=torch.float32, non_blocking=True)
            y = batch["output"].squeeze().to("cuda", dtype=torch.float32, non_blocking=True)
            pixdim = batch["pixdim"].squeeze()

            timesteps = torch.ones(x.shape[0]).to("cuda", dtype=torch.float32)

            ynet = net(x)
            loss = loss_fn(ynet, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 3 == 0 and b == 0:
                
                with torch.no_grad():
                    val_batch = next(iter(valid_loader))
                    x_val = val_batch["input"].squeeze(1).to("cuda", dtype=torch.float32, non_blocking=True)
                    y_val = val_batch["output"].squeeze(1).to("cuda", dtype=torch.float32, non_blocking=True)
                    pixdim = val_batch["pixdim"].squeeze()
                    ynet_val = net(x_val)
                    val_loss = loss_fn(ynet_val, y_val)
                    print(f"Validation Loss: {val_loss.item():.4f}")
                    
                    title = f"Epoch {epoch} Prediction"
                    kwargs = {"slices": (0.5, 0.5, 0.5), "thicknesses": (10, 10, 10), "vmin": -1, "vmax": 1}
                    plot_slices(ynet_val.squeeze(), pixdim, **kwargs)
                    plt.gca().set_title(title)
                    plt.gcf().savefig(f"training/unet/pred.png")
                    plot_slices(y_val.cpu().squeeze(), pixdim, **kwargs)
                    plt.gca().set_title(title)
                    plt.gcf().savefig(f"training/unet/out.png")
                    plot_slices(x_val.cpu().squeeze(), pixdim, **kwargs)
                    plt.gca().set_title(title)
                    plt.gcf().savefig(f"training/unet/inp.png")
                    plt.close('all')
                
                torch.save(net.state_dict(), f"training/unet/epoch_{epoch}.pt")
                
            
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            