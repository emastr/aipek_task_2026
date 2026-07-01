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
    from data_monai import NeighborFeature, create_data_loader, plot_slices
    
    epochs = 300
    start_epoch = 50
    size = 128
    channels = 64 # Number of slices per chunk; must be divisible by 32 for SwinUNETR
    batch_size = 9 # Number of chunks per GPU batch
    num_slices = 1
    neighbor_count = 1
    save_path = "training/unet_neighbor_big/"
    
    train_loader = create_data_loader(
        image_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_INP}",
        label_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_OUT}",
        channels = channels, # Number of slices per chunk
        batch_size = batch_size, # Number of chunks per GPU batch
        num_slices = num_slices, # Number of slices per chunk
        device = "cuda",
        size = size,
        random_flip = True,
        random_shift = True,
    )
    
    neighbor_loader = NeighborFeature(
        image_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_INP}",
        label_dir = f"{CONSTANTS.NORM_DATA_PATH_TRAIN_OUT}",
        channels = channels, # Number of slices per chunk
        neighbors = neighbor_count, # Number of nearest-neighbor feature maps
        num_slices = num_slices, # Number of slices per chunk
        device = "cuda",
        size = size,
        coarse_size = 32
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

    net = UNet(
        spatial_dims = 3, 
        in_channels = 2 + neighbor_count,
        out_channels = 1, 
        strides = (2, 2, 2),          # Downsampling factors
        channels = (64, 64, 128, 256), # old: (16, 32, 64, 128)  # Res: (128, 64, 32, 16)
        kernel_size=3, 
        up_kernel_size=3, 
        num_res_units=2,
        dropout=0.1
        ).to("cuda", dtype=torch.float32)

    # Load pretrained weights from the Swin Transformer model
    if start_epoch > 0:
        net.load_state_dict(torch.load(f"{save_path}epoch_{start_epoch}.pt"), strict=False)
    
    class ClampNet(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            
        def forward(self, x):
            return torch.tanh(self.net(x)) # clamp to [-1, 1]
    

    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )


    def weighted_l2(ynet, y):
        y_01 = (y + 1)/2
        ynet_01 = (ynet + 1)/2
        false_pos_weight =50.0
        false_neg_weight = 1.0
        both = y_01 #(y_01 + torch.clamp(ynet_01, 0, 1)) / 2
        loss_weight = (both * false_pos_weight + (1-both) * false_neg_weight)/(false_pos_weight + false_neg_weight)
        return torch.mean(loss_weight * (ynet -  y)**2)
    
    def weighted_l1(ynet, y):
        y_01 = (y + 1)/2
        ynet_01 = (ynet + 1)/2
        false_pos_weight =  50.0
        false_neg_weight = 1.0
        both = y_01 #(y_01 + torch.clamp(ynet_01, 0, 1)) / 2
        loss_weight = (both * false_pos_weight + (1-both) * false_neg_weight)/(false_pos_weight + false_neg_weight)
        return torch.mean(loss_weight * torch.abs(ynet -  y))
      
    def expectile_l2(ynet, y):
        diff = (ynet - y)/2
        false_pos_weight = 100.0
        false_neg_weight = 1.0
        expectile_weight = ((diff > 0) * false_pos_weight + (diff <= 0) * false_neg_weight)/(false_pos_weight + false_neg_weight)
        return torch.mean(expectile_weight * (ynet -  y)**2)
  
    loss_fn = weighted_l2  # or expectile_l2, depending on your preference
        
    def validation_plots(net, loader, path, epoch):
        iter_val = iter(loader)
        for i in range(3):
            val_batch = next(iter_val)
            x_val = val_batch["input"].to("cuda", dtype=torch.float32, non_blocking=True)
            y_val = val_batch["output"].to("cuda", dtype=torch.float32, non_blocking=True)
            pixdim = val_batch["pixdim"].squeeze()
            case_id_batch = val_batch.get("case_id", None)
            x_nei = neighbor_loader.get_neighbors(x_val, ignore_case_id=case_id_batch)
            x_val_aug = torch.cat((x_val, x_nei), dim=1)
            ynet_val = net(x_val_aug)
            val_loss = loss_fn(ynet_val, y_val)
            
            title = f"Epoch {epoch} Prediction"
            
            slices = (0.4, 0.4, 0.4)
            thicknes = (10, 10, 10)
            kwargs = {"vmin": -1, "vmax": 1, "cmap": "gray"}
            plot_slices(ynet_val[0, 0].cpu(), pixdim, slices, (1,1,1), **kwargs)
            plt.gca().set_title(title)
            plt.gcf().savefig(f"{path}pred{i}.png")
            plot_slices(y_val[0, 0].cpu(), pixdim, slices, thicknes, **kwargs)
            plt.gca().set_title(title)
            plt.gcf().savefig(f"{path}out{i}.png")
            plot_slices(x_val[0, 0].cpu(), pixdim, slices, thicknes, **kwargs)
            plt.gca().set_title(title)
            plt.gcf().savefig(f"{path}inp{i}.png")
            plot_slices(x_nei[0, 0].cpu(), pixdim, slices, thicknes, **kwargs)
            plt.gca().set_title(title)
            plt.gcf().savefig(f"{path}nei{i}.png")
            plt.close('all')
            
        plt.figure(figsize=(20, 10))
        for i in range(3):
            for j, target in enumerate(["pred", "out", "inp", "nei"]):
                plt.subplot(3, 4, i*4 + j + 1)
                plt.imshow(im.imread(f"{path}{target}{i}.png"))
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{path}examples.png")
        plt.close('all')
    
        print(f"Epoch {epoch} Validation Loss: {val_loss.item():.4f}")
        return val_loss.item()
    
    
    losses = []
    for epoch in range(start_epoch, start_epoch + epochs):
        for b, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            x = batch["input"].to("cuda", dtype=torch.float32, non_blocking=True)
            y = batch["output"].to("cuda", dtype=torch.float32, non_blocking=True)
            pixdim = batch["pixdim"].squeeze()
            case_id_batch = batch.get("case_id", None)

            x_nei = neighbor_loader.get_neighbors(x, ignore_case_id=case_id_batch)
            x_aug = torch.cat((x, x_nei), dim=1)
            x_aug = torch.flip(x_aug, dims=[2]) if torch.rand(1).item() < 0.5 else x_aug

            timesteps = torch.ones(x.shape[0]).to("cuda", dtype=torch.float32)

            ynet = net(x_aug)
            loss = loss_fn(ynet, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if epoch % 5 == 0 and b == 0:
                
                with torch.no_grad():
                    val_loss = validation_plots(net, valid_loader, f"{save_path}valid", epoch)
                    train_loss = validation_plots(net, trainval_loader, f"{save_path}train", epoch)
                    losses.append((epoch, val_loss, train_loss))
                    
                    plt.plot([e for e, _, _ in losses], [v for _, v, _ in losses], label="Validation Loss")
                    plt.plot([e for e, _, _ in losses], [t for _, _, t in losses], label="Train Loss")
                    plt.legend()
                    plt.savefig(f"{save_path}losses.png")
                    plt.close()
                    
                    
                if epoch % 50 == 0:
                    torch.save(net.state_dict(), f"{save_path}epoch_{epoch}.pt")
                
            
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, LR: {current_lr:.6e}")

        scheduler.step()
            