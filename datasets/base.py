import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """

    def __init__(
            self, 
            root_dir, 
            split='train', 
            downsample=1.0,
        ):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.poses)
    
    def to(self, device):
        self.rays = self.rays.to(device)
        self.poses = self.poses.to(device)
        self.K = self.K.to(device)
        self.directions = self.directions.to(device)
        if hasattr(self, 'depths'):
                self.depths = self.depths.to(device)
        return self

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images':  # randomly select images
                # img_idxs = np.random.choice(len(self.poses), self.batch_size)
                img_idxs = torch.randint(
                    0,
                    len(self.poses),
                    size=(self.batch_size,),
                    device=self.rays.device,
                )
            elif self.ray_sampling_strategy == 'same_image':  # randomly select ONE image
                # img_idxs = np.random.choice(len(self.poses), 1)[0]
                img_idxs = [idx]
            # randomly select pixels
            # pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1],
            #                             self.batch_size)
            
            if hasattr(self, 'pixel_sampling_strategy') and self.pixel_sampling_strategy=='entire_image':
                pix_idxs = torch.arange(
                    0, self.img_wh[0]*self.img_wh[1], device=self.rays.device
                )
            else:
                pix_idxs = torch.randint(
                    0, self.img_wh[0]*self.img_wh[1], size=(self.batch_size,), device=self.rays.device
                )


            rays = self.rays[img_idxs, pix_idxs]
            sample = {
                'img_idxs': img_idxs,
                'pix_idxs': pix_idxs,
                'pose': self.poses[img_idxs],
                'direction': self.directions[pix_idxs],
                'rgb': rays[:, :3],
            }
            if hasattr(self, 'depths'):
                sample['depth'] = self.depths[img_idxs, pix_idxs]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
             # if ground truth available
            if len(self.rays) > 0: 
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]

        return sample
