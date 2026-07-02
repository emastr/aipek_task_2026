

if __name__ == "__main__": # Avoids multiprocessing issues 
    import time
    import torch
    torch.manual_seed(0)

    import matplotlib.pyplot as plt
    from monai.networks.nets import UNet
    from geomloss import SamplesLoss
    import geomloss._legacy.sinkhorn_samples as _gl_sinkhorn_samples
    import geomloss._legacy.sinkhorn_divergence as _gl_sinkhorn_div

    from src.architectures import DURAG, NeighborLoader
    from src.plots import validation_plots
    from src.processing import CONSTANTS
    from src.data import create_data_loader

    def _softmin_tensorized_py314_fix(eps, C_xy, h_y):
        """
        GeomLoss 0.3.1 on Python 3.14 can pass ndarray-like values here.
        Ensure tensor semantics before reshape operations.
        """
        if not isinstance(h_y, torch.Tensor):
            h_y = torch.as_tensor(h_y, device=C_xy.device, dtype=C_xy.dtype)
        B = C_xy.shape[0]
        return -eps * (h_y.reshape(B, 1, -1) - C_xy / eps).logsumexp(2).reshape(B, -1)

    # Force sinkhorn_tensorized to use patched softmin implementation.
    _gl_sinkhorn_samples.softmin_tensorized = _softmin_tensorized_py314_fix
    _gl_sinkhorn_samples.sinkhorn_tensorized.__globals__["softmin_tensorized"] = _softmin_tensorized_py314_fix

    _orig_sinkhorn_loop = _gl_sinkhorn_div.sinkhorn_loop
    def _sinkhorn_loop_tensor_safe(softmin, *args, **kwargs):
        """
        Ensure every softmin output stays a torch.Tensor.
        GeomLoss legacy can occasionally propagate numpy scalars/arrays in Py3.14.
        """
        def _softmin_safe(eps, C_xy, h_y):
            out = softmin(eps, C_xy, h_y)
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, device=C_xy.device, dtype=C_xy.dtype)
            return out

        return _orig_sinkhorn_loop(_softmin_safe, *args, **kwargs)

    _gl_sinkhorn_div.sinkhorn_loop = _sinkhorn_loop_tensor_safe
    _gl_sinkhorn_samples.sinkhorn_loop = _sinkhorn_loop_tensor_safe
    _gl_sinkhorn_samples.sinkhorn_tensorized.__globals__["sinkhorn_loop"] = _sinkhorn_loop_tensor_safe

    epochs = 1000
    start_epoch = 0
    size = 128
    channels = 64 # Number of slices per chunk; must be divisible by 32 for SwinUNETR
    batch_size = 16 # Number of chunks per GPU batch
    neighbor_count = 3
    save_path = "training/durag_wasserstein/"
    wasserstein_points = 1024

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

    # Sinkhorn divergence between two sampled point clouds.
    sinkhorn_loss = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=0.05,
        scaling=0.9,
        debias=False,
        backend="tensorized",
    )


    def loss_fn(ynet, y, pixdim=None):
        y_01 = torch.sigmoid(y)
        ynet_01 = torch.sigmoid(ynet)
        return wasserstein(ynet_01, y_01, pixdim=pixdim)

        
    def _build_coords(spatial_shape, device, dtype):
        axes = [torch.linspace(0.0, 1.0, steps=s, device=device, dtype=dtype) for s in spatial_shape]
        mesh = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(mesh, dim=-1).reshape(-1, len(spatial_shape))


    def _sample_point_cloud(value_map, num_points=1024, pixdim=None):
        """
        Draw an intensity-weighted point cloud from a normalized value map.
        """
        b = value_map.shape[0]
        spatial_shape = value_map.shape[1:]
        coords = _build_coords(spatial_shape, value_map.device, value_map.dtype)
        flat_values = value_map.reshape(b, -1).clamp(min=0.0, max=1.0)

        # Fallback to uniform sampling if a map is all zeros.
        weights = flat_values + 1e-8
        weights = weights / weights.sum(dim=1, keepdim=True)

        sampled_idx = torch.multinomial(weights, num_samples=num_points, replacement=True)

        d = coords.shape[1]
        coords_expanded = coords.unsqueeze(0).expand(b, -1, -1)
        idx_expanded = sampled_idx.unsqueeze(-1).expand(-1, -1, d)
        sampled_coords = torch.gather(coords_expanded, 1, idx_expanded)

        sampled_weights = torch.gather(flat_values, 1, sampled_idx) + 1e-8
        sampled_weights = sampled_weights / sampled_weights.sum(dim=1, keepdim=True)

        if pixdim is not None:
            if pixdim.ndim == 1:
                sampled_coords = sampled_coords * pixdim[:d].to(sampled_coords.device, sampled_coords.dtype).view(1, 1, d)
            elif pixdim.ndim == 2:
                sampled_coords = sampled_coords * pixdim[:, :d].to(sampled_coords.device, sampled_coords.dtype).view(b, 1, d)

        # GeomLoss legacy internals rely on view(); enforce contiguous layout.
        return sampled_coords.contiguous(), sampled_weights.contiguous()


    def _memory_error(err):
        msg = str(err)
        return (
            "CUBLAS_STATUS_ALLOC_FAILED" in msg
            or "CUDA out of memory" in msg
            or "out of memory" in msg
        )


    def _sinkhorn_safe(a_i, x_i, b_i, y_i, min_points=128):
        """
        Robust GeomLoss call: retry with fewer points on CUDA memory pressure.
        """
        n = a_i.shape[0]
        last_err = None

        while n >= min_points:
            idx_a = torch.randperm(a_i.shape[0], device=a_i.device)[:n]
            idx_b = torch.randperm(b_i.shape[0], device=b_i.device)[:n]

            a_n = a_i[idx_a]
            b_n = b_i[idx_b]
            a_n = a_n / (a_n.sum() + 1e-8)
            b_n = b_n / (b_n.sum() + 1e-8)
            x_n = x_i[idx_a]
            y_n = y_i[idx_b]

            try:
                return sinkhorn_loss(a_n, x_n, b_n, y_n)
            except RuntimeError as err:
                if not _memory_error(err):
                    raise
                last_err = err
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                n //= 2

        # Last-resort CPU fallback to keep training alive.
        a_cpu = a_i.to("cpu")
        x_cpu = x_i.to("cpu")
        b_cpu = b_i.to("cpu")
        y_cpu = y_i.to("cpu")
        try:
            return sinkhorn_loss(a_cpu, x_cpu, b_cpu, y_cpu).to(x_i.device)
        except RuntimeError:
            if last_err is not None:
                raise last_err
            raise


    def wasserstein(y_pred, y_true, pixdim=None):
        """
        Subsample a point cloud from y_pred and y_true, then compute the Wasserstein distance between them.
        """
        # Collapse channel axis to one normalized scalar field per sample.
        pred_map = y_pred.mean(dim=1)
        true_map = y_true.mean(dim=1)

        pred_pts, pred_w = _sample_point_cloud(pred_map, num_points=wasserstein_points, pixdim=pixdim)
        true_pts, true_w = _sample_point_cloud(true_map, num_points=wasserstein_points, pixdim=pixdim)

        # GeomLoss batched legacy path can fail on some Python/GeomLoss combos.
        # Compute per-sample losses to keep execution on the robust non-batched path.
        batch_losses = []
        for i in range(pred_pts.shape[0]):
            a_i = pred_w[i].contiguous()
            x_i = pred_pts[i].contiguous()
            b_i = true_w[i].contiguous()
            y_i = true_pts[i].contiguous()
            batch_losses.append(
                _sinkhorn_safe(a_i, x_i, b_i, y_i)
            )
        return torch.stack(batch_losses).mean()
        
        
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
            loss = loss_fn(ynet, y, pixdim=pixdim)
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
            