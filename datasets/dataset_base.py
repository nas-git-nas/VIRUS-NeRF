import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd

from args.args import Args



class DatasetBase(Dataset):
    """
    Define length and sampling method
    """

    def __init__(
            self, 
            args:Args, 
            split='train', 
        ):
        self.args = args
        self.split = split

    def read_intrinsics(self):
        pass

    def __len__(self):
        return len(self.poses)
    
    def __call__(
        self, 
        batch_size:int,
        sampling_strategy:dict,
        origin:str,
    ):
        """
        Get some data from the dataset.
        Args:
            batch_size: number of samples; int
            sampling_strategy: dictionary containing the sampling strategy for images and pixels; dict
            origin: sampling origin; str
                    'nerf': sample for nerf
                    'occ': sample for occupancy grid
        """
        # sample image and pixel indices
        img_idxs, pix_idxs, count = self.sampler(
            batch_size=batch_size,
            sampling_strategy=sampling_strategy,
            origin=origin,
            stack_ids=None,
        )

        # sample data
        rays = self.rays[img_idxs, pix_idxs]
        samples = {
            'img_idxs': img_idxs,
            'pix_idxs': pix_idxs,
            'sample_count': count.detach().clone(),
            'pose': self.poses[img_idxs].detach().clone().requires_grad_(True),
            'direction': self.directions[pix_idxs].detach().clone().requires_grad_(True),
            'rgb': rays[:, :3].detach().clone().requires_grad_(True),
        }
        if hasattr(self, 'depths_dict'):
            samples['depth'] = {}
            for sensor, sensor_depths in self.depths_dict.items():
                samples['depth'][sensor] = sensor_depths[img_idxs, pix_idxs].detach().clone().requires_grad_(True)

        return samples
    
    def to(self, device):
        self.rays = self.rays.to(device)
        self.poses = self.poses.to(device)
        self.K = self.K.to(device)
        self.directions = self.directions.to(device)
        self.times = self.times.to(device)
        if hasattr(self, 'depths_dict'):
            for key in self.depths_dict.keys():
                self.depths_dict[key] = self.depths_dict[key].to(device)
        return self
    
    def getMeanHeight(
        self,
    ):
        """
        Get mean height of the images.
        Returns:
            mean_height: mean height of the images; float
        """
        mean_height = torch.mean(self.poses[:, 2, 3])
        return mean_height.item()
    
    def getValidDepthMask(
        self,
        img_idxs:torch.Tensor,
    ):
        """
        Get valid depth masks for each sensor.
        Args:
            img_idxs: indices of images; tensor of int64 (batch_size,)
        Returns:
            val_depth_masks: valid depth masks for each sensor; dict of tensors of bool (batch_size, H*W)
        """
        val_depth_masks = {}
        for sensor, sensor_depths in self.depths_dict.items():
            val_depth_masks[sensor] = ~torch.isnan(sensor_depths[img_idxs])
        return val_depth_masks
    
    def reduceImgHeight(
        self,
        rays:torch.Tensor,
        directions:torch.Tensor,
        depths:np.array,
        img_wh:tuple,
        angle_min_max:tuple,
    ):
        """
        Reduce the image height to the specified range.
        Args:
            rays: rays; tensor of shape (N, H*W, 3)
            directions: ray directions; tensor of shape (H*W, 3)
            depths: depths; numpy array of shape (N, H*W)
            img_wh: image width and height; tuple of ints
            angle_min_max: tuple containing the min and max angles of the image to keep
        Returns:
            rays: rays; tensor of shape (N, H*W, 3)
            directions: ray directions; tensor of shape (H*W, 3)
            depths: depths; tensor of shape (N, H*W)
            img_wh: image width and height; tuple of ints
        """
        rays = rays.clone().detach()
        directions = directions.clone().detach()
        depths = np.copy(depths)
        W, H = img_wh
        N = rays.shape[0]

        # verify dimensions and reshape tensors
        if rays.shape[0] != depths.shape[0]:
            self.args.logger.error(f"rays and depths must have the same number of images")
        if rays.shape[1] != W*H or directions.shape[0] != W*H or depths.shape[1] != W*H:
            self.args.logger.error(f"rays, directions and depths must have the same number of pixels = {W*H}")
        rays = rays.reshape(N, H, W, 3)
        directions = directions.reshape(H, W, 3)
        depths = depths.reshape(N, H, W)

        # convert angles to indices
        idx_slope = H / self.args.rgbd.angle_of_view[1]
        idx_min_max = (
            max(np.floor(H/2 + idx_slope*angle_min_max[0]).astype(int), 0),
            min(np.ceil(H/2 + idx_slope*angle_min_max[1]).astype(int), H),
        )
        print(f"idx_min_max: {idx_min_max}")

        # reduce image height
        img_wh = (W, idx_min_max[1]-idx_min_max[0])
        rays = rays[:, idx_min_max[0]:idx_min_max[1], :, :]
        directions = directions[idx_min_max[0]:idx_min_max[1], :, :]
        depths = depths[:, idx_min_max[0]:idx_min_max[1], :]

        # reshape tensors
        rays = rays.reshape(N, img_wh[0]*img_wh[1], 3)
        directions = directions.reshape(img_wh[0]*img_wh[1], 3)
        depths = depths.reshape(N, img_wh[0]*img_wh[1])
        return rays, directions, depths, img_wh
    
    def splitDataset(self, df, split_ratio, split_description_path, split_description_name):
        """
        Split the dataset into train, val and test sets.
        Args:
            df: dataframe containing the dataset
            split_ratio: dictionary containing the split ratio for each split
            split_description_path: path to the directory containing the split description; str
            split_description_name: filename of split description; str
        Returns:
            df: dataframe containing the dataset with a new column 'split'
        """
        df = df.copy(deep=True) 

        # load split description if it exists already
        df_description = None
        if os.path.exists(os.path.join(split_description_path, 'split_description.csv')):    
            df_description = pd.read_csv(os.path.join(split_description_path, 'split_description.csv'), 
                                         index_col=0, dtype={'info':str,'train':float, 'val':float, 'test':float})
        
        # load split if it exists already
        if os.path.exists(os.path.join(split_description_path, split_description_name)):
            # split ratio must be the same as in description (last split)
            if df_description.loc[split_description_name, 'train']==split_ratio['train'] \
                and df_description.loc[split_description_name, 'val']==split_ratio['val'] \
                and df_description.loc[split_description_name, 'test']==split_ratio['test']:

                # load split and merge with df
                df_split = pd.read_csv(os.path.join(split_description_path, split_description_name))
                df = pd.merge(df, df_split, on='id', how='left')
                return df

        # verify that split is correct
        if split_ratio['train'] + split_ratio['val'] + split_ratio['test'] != 1.0:
            self.args.logger.error(f"split ratios do not sum up to 1.0")
        if split_ratio['train']*10 % 1 != 0 or split_ratio['val']*10 % 1 != 0 or split_ratio['test']*10 % 1 != 0:
            self.args.logger.error(f"split ratios must be multiples of 0.1")
        
        # get indices for each sensor
        split_idxs = {"train": np.empty(0, dtype=int), "val": np.empty(0, dtype=int), "test": np.empty(0, dtype=int)}
        for id in df["sensor_id"].unique():
            id_idxs = df.index[df["sensor_id"] == id].to_numpy()
   
            # get indices for each split
            partitions = ["train" for _ in range(int(split_ratio['train']*10))] \
                        + ["val" for _ in range(int(split_ratio['val']*10))] \
                        + ["test" for _ in range(int(split_ratio['test']*10))]
            for offset, part in enumerate(partitions):
                split_idxs[part] = np.concatenate((split_idxs[part], id_idxs[offset::10]))

        # assign split
        df.insert(1, 'split', None) # create new column for split
        df.loc[split_idxs["train"], 'split'] = 'train'
        df.loc[split_idxs["val"], 'split'] = 'val'
        df.loc[split_idxs["test"], 'split'] = 'test'

        # save split
        df_split = df[['id', 'split', 'sensor_name']].copy(deep=True)
        df_split.to_csv(os.path.join(split_description_path, split_description_name), index=False)

        # save split description
        if df_description is None:
            df_description = pd.DataFrame(columns=['info','train', 'val', 'test'])
            df_description.loc["info"] = ["This file contains the split ratios for each split file in the same directory. " \
                                          + "The Ratios must be a multiple of 0.1 and sum up to 1.0 to ensure correct splitting.", "", "", ""]
        df_description.loc[split_description_name] = ["-", split_ratio['train'], split_ratio['val'], split_ratio['test']]
        df_description.to_csv(os.path.join(split_description_path, 'split_description.csv'), index=True)

        return df




# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import os
# import pandas as pd

# from datasets.ray_utils import get_rays

# from args.args import Args



# class DatasetBase(Dataset):
#     """
#     Define length and sampling method
#     """

#     def __init__(
#             self, 
#             args:Args, 
#             split='train', 
#         ):
#         self.args = args
#         self.split = split

#     def read_intrinsics(self):
#         pass

#     def __len__(self):
#         return len(self.poses)
    
#     def __call__(
#         self, 
#         batch_size:int,
#         sampling_strategy:dict,
#         origin:str,
#     ):
#         """
#         Get some data from the dataset.
#         Args:
#             batch_size: number of samples; int
#             sampling_strategy: dictionary containing the sampling strategy for images and pixels; dict
#             origin: sampling origin; str
#                     'nerf': sample for nerf
#                     'occ': sample for occupancy grid
#         Returns:
#             samples: dictionary containing the sampled data; dict
#         """
#         # sample image and pixel indices
#         img_idxs, pix_idxs, count = self.sampler(
#             batch_size=batch_size,
#             sampling_strategy=sampling_strategy,
#             stack_ids=self.stack_ids,
#             origin=origin,
#         )

#         # calculate ray origins and directions
#         rays_o, rays_d = self._calcRayPoses(
#             directions_dict=self.directions_dict,
#             poses=self.poses,
#             stack_ids=self.stack_ids,
#             img_idxs=img_idxs,
#             pix_idxs=pix_idxs,
#         )

#         # sample data

#         poses = self.poses[img_idxs]
#         directions = self.directions_dict['RGBD_1'][pix_idxs]

#         stack_ids = self.stack_ids[img_idxs]
#         rgbs = self.rgbs[img_idxs, pix_idxs, :3]
#         samples = {
#             'img_idxs': img_idxs,
#             'pix_idxs': pix_idxs,
#             'sample_count': count.detach().clone(),
#             'stack_id': stack_ids.detach().clone().requires_grad_(False),
#             'rays_o': rays_o.detach().clone().requires_grad_(True),
#             'rays_d': rays_d.detach().clone().requires_grad_(True),
#             'pose': poses.detach().clone().requires_grad_(True),
#             'direction': directions.detach().clone().requires_grad_(True),
#             'rgb': rgbs.detach().clone().requires_grad_(True),
#             'depth': {},
#         }
#         for sensor, sensor_depths in self.depths_dict.items():
#             samples['depth'][sensor] = sensor_depths[img_idxs, pix_idxs].detach().clone().requires_grad_(True)

#         return samples
    
#     def to(self, device):
#         self.rgbs = self.rgbs.to(device)
#         self.poses = self.poses.to(device)
#         # self.Ks = self.Ks.to(device)
#         self.times = self.times.to(device)
#         self.stack_ids = self.stack_ids.to(device)
#         for key in self.depths_dict.keys():
#             self.depths_dict[key] = self.depths_dict[key].to(device)
#         for cam_id, directions in self.directions_dict.items():
#             self.directions_dict[cam_id] = directions.to(device)
#         return self
    
#     def getMeanHeight(
#         self,
#     ):
#         """
#         Get mean height of the images.
#         Returns:
#             mean_height: mean height of the images; float
#         """
#         mean_height = torch.mean(self.poses[:, 2, 3])
#         return mean_height.item()
    
#     def getValidDepthMask(
#         self,
#         img_idxs:torch.Tensor,
#     ):
#         """
#         Get valid depth masks for each sensor.
#         Args:
#             img_idxs: indices of images; tensor of int64 (batch_size,)
#         Returns:
#             val_depth_masks: valid depth masks for each sensor; dict of tensors of bool (batch_size, H*W)
#         """
#         val_depth_masks = {}
#         for sensor, sensor_depths in self.depths_dict.items():
#             val_depth_masks[sensor] = ~torch.isnan(sensor_depths[img_idxs])
#         return val_depth_masks
    
#     def reduceImgHeight(
#         self,
#         rgbs:torch.Tensor,
#         directions:torch.Tensor,
#         depths:np.array,
#         img_wh:tuple,
#         angle_min_max:tuple,
#     ):
#         """
#         Reduce the image height to the specified range.
#         Args:
#             rgbs: colors; tensor of shape (N, H*W, 3)
#             directions: ray directions; tensor of shape (H*W, 3)
#             depths: depths; numpy array of shape (N, H*W)
#             img_wh: image width and height; tuple of ints
#             angle_min_max: tuple containing the min and max angles of the image to keep
#         Returns:
#             rgbs: colors; tensor of shape (N, H*W, 3)
#             directions: ray directions; tensor of shape (H*W, 3)
#             depths: depths; tensor of shape (N, H*W)
#             img_wh: image width and height; tuple of ints
#         """
#         rgbs = rgbs.clone().detach()
#         directions = directions.clone().detach()
#         depths = np.copy(depths)
#         W, H = img_wh
#         N = rgbs.shape[0]

#         # verify dimensions and reshape tensors
#         if rgbs.shape[0] != depths.shape[0]:
#             self.args.logger.error(f"rgbs and depths must have the same number of images")
#         if rgbs.shape[1] != W*H or directions.shape[0] != W*H or depths.shape[1] != W*H:
#             self.args.logger.error(f"rgbs, directions and depths must have the same number of pixels = {W*H}")
#         rgbs = rgbs.reshape(N, H, W, 3)
#         directions = directions.reshape(H, W, 3)
#         depths = depths.reshape(N, H, W)

#         # convert angles to indices
#         idx_slope = H / self.args.rgbd.angle_of_view[1]
#         idx_min_max = (
#             max(np.floor(H/2 + idx_slope*angle_min_max[0]).astype(int), 0),
#             min(np.ceil(H/2 + idx_slope*angle_min_max[1]).astype(int), H),
#         )
#         print(f"idx_min_max: {idx_min_max}")

#         # reduce image height
#         img_wh = (W, idx_min_max[1]-idx_min_max[0])
#         rgbs = rgbs[:, idx_min_max[0]:idx_min_max[1], :, :]
#         directions = directions[idx_min_max[0]:idx_min_max[1], :, :]
#         depths = depths[:, idx_min_max[0]:idx_min_max[1], :]

#         # reshape tensors
#         rgbs = rgbs.reshape(N, img_wh[0]*img_wh[1], 3)
#         directions = directions.reshape(img_wh[0]*img_wh[1], 3)
#         depths = depths.reshape(N, img_wh[0]*img_wh[1])
#         return rgbs, directions, depths, img_wh
    
#     def _calcRayPoses(
#         self,
#         directions_dict:torch.Tensor,
#         poses:torch.Tensor,
#         stack_ids:torch.Tensor,
#         img_idxs:torch.Tensor,
#         pix_idxs:torch.Tensor,
#     ):
#         """
#         Calculate ray origins and directions for a batch of rays.
#         Args:
#             directions_dict: dictionary containing the directions for each sensor; dict { cam_id: directions (H*W, 3) }
#             poses: poses; tensor of shape (N_dataset, 3, 4)
#             stack_ids: stack ids; tensor of int64 (N_dataset,)
#             img_idxs: indices of images; tensor of int64 (N_batch,)
#             pix_idxs: indices of pixels; tensor of int64 (N_batch,)
#         Returns:
#             rays_o: ray origins; tensor of shape (N_batch, 3)
#             rays_d: ray directions; tensor of shape (N_batch, 3)
#         """
#         N = img_idxs.shape[0]

#         rays_o = torch.full((N, 3), np.nan, dtype=torch.float32, device=self.args.device)
#         rays_d = torch.full((N, 3), np.nan, dtype=torch.float32, device=self.args.device)
#         for cam_id, directions in directions_dict.items():

#             idx_mask = (stack_ids[img_idxs] == int(cam_id[-1])) # (N,)
#             img_idxs_temp = img_idxs[idx_mask] # (n,)
#             pix_idxs_temp = pix_idxs[idx_mask] # (n,)
    
#             rays_o_temp, rays_d_tempt = get_rays(
#                 directions=directions[pix_idxs_temp],
#                 c2w=poses[img_idxs_temp],
#             ) # (n, 3), (n, 3)

#             rays_o[idx_mask] = rays_o_temp # (N, 3)
#             rays_d[idx_mask] = rays_d_tempt # (N, 3)

#         if self.args.model.debug_mode:
#             if torch.any(torch.isnan(rays_o)) or torch.any(torch.isnan(rays_d)):
#                 self.args.logger.error(f"DatasetBase:_calcRayPoses: some rays were not calculated correctly")

#         return rays_o, rays_d





            # # training pose is retrieved in train.py
            # if self.args.training.sampling_strategy["imgs"] == "all":
            #     img_idxs = torch.randint(0, len(self.poses), size=(self.args.training.batch_size,), device=self.rays.device)
            # elif self.args.training.sampling_strategy["imgs"] == "same":
            #     img_idxs = idx * torch.ones(self.args.training.batch_size, dtype=torch.int64, device=self.rays.device)               
            # else:
            #     print(f"ERROR: base.py: __getitem__: image sampling strategy must be either 'all' or 'same' " \
            #           f"but is {self.args.training.sampling_strategy['imgs']}")

            # # if self.ray_sampling_strategy == 'all_images':  # randomly select images
            # #     # img_idxs = np.random.choice(len(self.poses), self.batch_size)
            # #     img_idxs = torch.randint(
            # #         0,
            # #         len(self.poses),
            # #         size=(self.batch_size,),
            # #         device=self.rays.device,
            # #     )
            # # elif self.ray_sampling_strategy == 'same_image':  # randomly select ONE image
            # #     # img_idxs = np.random.choice(len(self.poses), 1)[0]
            # #     img_idxs = [idx]
            # # # randomly select pixels
            # # # pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1],
            # # #                             self.batch_size)

            # if self.args.training.sampling_strategy["rays"] == "random":
            #     pix_idxs = torch.randint(0, self.img_wh[0]*self.img_wh[1], size=(self.args.training.batch_size,), device=self.rays.device)
            # elif self.args.training.sampling_strategy["rays"] == "ordered":
            #     step = self.img_wh[0]*self.img_wh[1] / self.args.training.batch_size
            #     pix_idxs = torch.linspace(0, self.img_wh[0]*self.img_wh[1]-1-step, self.args.training.batch_size, device=self.rays.device)
            #     rand_offset = step * torch.rand(size=(self.args.training.batch_size,), device=self.rays.device)
            #     pix_idxs = torch.round(pix_idxs + rand_offset).to(torch.int64)
            #     pix_idxs = torch.clamp(pix_idxs, min=0, max=self.img_wh[0]*self.img_wh[1]-1)
            # elif self.args.training.sampling_strategy["rays"] == "closest":
            #     pix_idxs = torch.randint(0, self.img_wh[0]*self.img_wh[1], size=(self.args.training.batch_size,), device=self.rays.device)
            #     num_min_idxs = int(0.005 * self.args.training.batch_size)
            #     pix_min_idxs = self.sensors_dict["USS"].imgs_min_idx
            #     pix_idxs[:num_min_idxs] = pix_min_idxs[img_idxs[:num_min_idxs]]
            # else:
            #     print(f"ERROR: base.py: __getitem__: pixel sampling strategy must be either 'random' or 'ordered' " \
            #           f"but is {self.args.training.sampling_strategy['pixels']}")
            
            # # if hasattr(self, 'pixel_sampling_strategy') and self.pixel_sampling_strategy=='entire_image':
            # #     pix_idxs = torch.arange(
            # #         0, self.img_wh[0]*self.img_wh[1], device=self.rays.device
            # #     )
            # # else:
            # #     pix_idxs = torch.randint(
            # #         0, self.img_wh[0]*self.img_wh[1], size=(self.batch_size,), device=self.rays.device
            # #     )



            #         else:
            # sample = {
            #     'pose': self.poses[idx], 
            #     'img_idxs': idx
            # }
            # if hasattr(self, 'depths_dict'):
            #     sample['depth'] = {}
            #     for sensor, sensor_depths in self.depths_dict.items():
            #         sample['depth'][sensor] = sensor_depths[idx]
            #  # if ground truth available
            # if len(self.rays) > 0: 
            #     rays = self.rays[idx]
            #     sample['rgb'] = rays[:, :3]