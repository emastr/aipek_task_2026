if __name__ == "__main__":
    import sys
    sys.path.append('')
    sys.path.append('../')


    import torch
    torch.manual_seed(0)
    import matplotlib.image as im
    import matplotlib.pyplot as plt
    from data import CONSTANTS
    from monai.networks.nets import UNet
    from data_monai import create_data_loader, plot_slices
    
    epochs = 3000
    size = 128
    channels = 8 # Number of slices per chunk
    batch_size = 12 # Number of chunks per GPU batch
    num_slices = 20
    save_path = "training/unet_res_weighted_2/"
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

    unet = UNet(
        spatial_dims=2,
        in_channels=channels * 2,
        out_channels=channels,
        channels=(64, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to("cuda", dtype=torch.float32)
    
    class ClampNet(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            
        def forward(self, x):
            return torch.tanh(self.net(x)) # clamp to [-1, 1]
    
    net = unet# ClampNet(unet).to("cuda", dtype=torch.float32)
    
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )


    def weighted_l2(ynet, y):
        y_01 = (y + 1)/2
        ynet_01 = (ynet + 1)/2
        false_pos_weight = 100.0
        false_neg_weight = 1.0
        both = (y_01 + torch.clamp(ynet_01, 0, 1)) / 2
        loss_weight = (both * false_pos_weight + (1-both) * false_neg_weight)/(false_pos_weight + false_neg_weight)
        return torch.mean(loss_weight * (ynet -  y)**2)
    
    def expectile_l2(ynet, y):
        diff = (ynet - y)/2
        false_pos_weight = 100.0
        false_neg_weight = 1.0
        expectile_weight = ((diff > 0) * false_pos_weight + (diff <= 0) * false_neg_weight)/(false_pos_weight + false_neg_weight)
        return torch.mean(expectile_weight * (ynet -  y)**2)
  
    loss_fn = weighted_l2  # or expectile_l2, depending on your preference
        
    for epoch in range(epochs):
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
                if epoch % 50 == 0:
                    torch.save(net.state_dict(), f"{save_path}epoch_{epoch}.pt")
                
            
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, LR: {current_lr:.6e}")

        scheduler.step()
            