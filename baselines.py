import torch
from torch import nn
import os
import sys

sys.path.append("../")
from data import NiiPoint
import torch.nn.functional as F



class NiiKNN(nn.Module):
    
    def __init__(self, k, img_path, label_path, max_num=None):
        super().__init__()
        self.k = k
        self.label_path = label_path
        self.img_path = img_path
        self.max_num = max_num
        
    def forward(self, x: NiiPoint):
        files = os.listdir(self.img_path)
        # read dataset.json to get size
        num_data = len(files)
        dist = torch.zeros(num_data, device="cuda")
        for i,file in enumerate(files):
            print(f"Calculating distance for {file}...", end="\r")
            img = NiiPoint.from_path(os.path.join(self.img_path, file), "cuda")
            diff = x - img
            dist[i] = torch.mean(diff.data**2)**0.5
            if self.max_num is not None and i >= self.max_num:
                break
            
        sorted_idx = torch.argsort(dist)
        for i in range(self.k):
            idx = sorted_idx[i]
            file = files[idx]
            label_file = f"{file[:9]}_0001.nii.gz"
            xi = NiiPoint.from_path(os.path.join(self.img_path, file), "cuda")
            yi = NiiPoint.from_path(os.path.join(self.label_path, label_file), "cuda")    
            if i == 0:
                x = xi * (1 / self.k)
                y = yi * (1 / self.k)
            else:
                x = x + xi * (1 / self.k)
                y = y + yi * (1 / self.k)
        return x, y
    

class NiiCKNN(nn.Module):
    
    def __init__(self, k, kernel_size, img_path, label_path, max_num=None, max_batch=None):
        super().__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.label_path = label_path
        self.img_path = img_path
        self.max_num = max_num
        self.max_batch = max_batch
        self.case_ids = [file[:9] for file in os.listdir(self.img_path)]
        self.data_size = len(self.case_ids)
    
    def sliding_l2(self, x_pt: torch.Tensor, pixdims: tuple):
        kernel_size = [2 * int(self.kernel_size[i] / pixdims[i]) + 1 for i in range(3)]
        pad = [int((kernel_size[i] - 1) / 2) for i in range(3)]
        padding = (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0])
        x_pt = x_pt ** 2
        x_pt = F.pad(x_pt.unsqueeze(1), padding, mode='constant', value=0.0)
        weight = torch.ones((1, 1,) + tuple(kernel_size), dtype=x_pt.dtype, device=x_pt.device)
        sliding_sum = F.conv3d(x_pt, weight, stride=1)
        return sliding_sum.squeeze(1)

    
    def forward(self, x: NiiPoint):
        with torch.no_grad():
            files = os.listdir(self.img_path)
            # read dataset.json to get size
            sqdist = torch.full(x.data.shape, torch.inf, device="cuda")                     # track closest block distance
            x_est = torch.zeros_like(x.data, device="cuda")                                   # track closest block image
            y_est = torch.zeros_like(x.data, device="cuda")  
            
            
            for b in range(min(self.max_batch, len(files) // self.max_num)):
                img = torch.zeros((self.max_num, ) + x.data.shape, device="cuda")
                lab = torch.zeros((self.max_num, ) + x.data.shape, device="cuda")
                
                print(f"Processing batch {b + 1}/{min(self.max_batch, len(files) // self.max_num)}...", end="\r")
                for i in range(self.max_num):
                    xi, yi = self.load_case(i + b * self.max_num)
                    img[i, :, :, :] = xi.interpolate_to(x.data.shape).data
                    lab[i, :, :, :] = yi.interpolate_to(x.data.shape).data
                
                
                sqdists_all = self.sliding_l2(
                    x.data[None, :, :, :] - img, 
                    pixdims = x.header.get_zooms()
                    )
                
                sqdist_new, args = torch.min(sqdists_all, dim=0)
                x_new = torch.gather(img, 0, args[None, :, :, :]).squeeze(0)
                y_new = torch.gather(lab, 0, args[None, :, :, :]).squeeze(0)
                
                x_est = torch.where(sqdist_new < sqdist, x_new, x_est)
                y_est = torch.where(sqdist_new < sqdist, y_new, y_est)
                sqdist = torch.min(sqdist, sqdist_new)
                if self.max_num is not None and i >= self.max_num:
                    break
        
            return NiiPoint(x_est, x.affine, x.header), NiiPoint(y_est, x.affine, x.header)

    def load_case(self, idx):
        assert idx < self.data_size, f"Index {idx} out of range for dataset of size {self.data_size}"
        case_id = self.case_ids[idx]
        img_file = f"{case_id}_0000.nii.gz"
        label_file = f"{case_id}_0001.nii.gz"
        img = NiiPoint.from_path(os.path.join(self.img_path, img_file), "cuda")
        label = NiiPoint.from_path(os.path.join(self.label_path, label_file), "cuda")    
        return img, label