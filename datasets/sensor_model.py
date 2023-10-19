import numpy as np
import torch
import matplotlib.pyplot as plt
from abc import abstractmethod
import skimage.measure
from typing import TypedDict

from args.args import Args


class SensorModel():
    def __init__(self, args:Args, img_wh:tuple) -> None:
        self.args = args
        self.W = img_wh[0]
        self.H = img_wh[1]
        
    @abstractmethod
    def convertDepth(self, depths):
        pass

    def pos2idx(self, pos_h, pos_w):
        """
        Convert position to index.
        Args:
            pos_h: position; array of shape (N,)
            pos_w: position; array of shape (N,)
        Returns:
            idxs_h: index; array of shape (N,)
            idxs_w: index; array of shape (N,)
        """
        idxs_h = None
        if pos_h is not None:
            idxs_h = np.round(pos_h).astype(int)
            idxs_h = np.clip(idxs_h, 0, self.H-1)

        idxs_w = None
        if pos_w is not None:
            idxs_w = np.round(pos_w).astype(int)
            idxs_w = np.clip(idxs_w, 0, self.W-1)

        return idxs_h, idxs_w

    def AoV2pixel(self, aov_sensor:list):
        """
        Convert the angle of view to width in pixels
        Args:
            aov_sensor: angle of view of sensor in width and hight; list
        Returns:
            num_pixels: width in pixels; int
        """
        num_pixels = np.min((self.W, self.H)) * np.min(aov_sensor) / np.min(self.args.rgbd.angle_of_view)
        return np.round(num_pixels).astype(int)
    

class RGBDModel(SensorModel):
    def __init__(self, args:Args, img_wh:tuple) -> None:
        """
        Sensor model for Time of Flight (ToF) sensor.
        Args:
            img_wh: image width and height, tuple of int
        """
        SensorModel.__init__(self, args, img_wh)     

    def convertDepth(self, depths):
        """
        Convert depth img using ToF sensor model. Set all unknown depths to nan.
        Args:
            depths: depth img; array of shape (N, H*W)
        Returns:
            depths: depth img converted to ToF sensor array; array of shape (N, H*W)
        """
        return np.copy(depths)


class ToFModel(SensorModel):
    def __init__(self, args:Args, img_wh:tuple) -> None:
        """
        Sensor model for Time of Flight (ToF) sensor.
        Args:
            img_wh: image width and height, tuple of int
        """
        SensorModel.__init__(self, args, img_wh)      

        # calculate indices of ToF sensor array
        width = self.AoV2pixel(aov_sensor=args.tof.angle_of_view)
        idxs_w = np.linspace(0, width, args.tof.matrix[0], dtype=float)
        idxs_h = np.linspace(0, width, args.tof.matrix[1], dtype=float)

        # ajust indices to quadratic shape
        idxs_w = idxs_w + (self.W - width)/2
        idxs_h = idxs_h + (self.H - width)/2

        # convert indices to int
        idxs_h, idxs_w = self.pos2idx(idxs_h, idxs_w)

        # create meshgrid of indices
        idxs_h, idxs_w = np.meshgrid(idxs_h, idxs_w, indexing='ij') # (H, W)
        self.idxs_h = idxs_h.flatten() # (H*W,)
        self.idxs_w = idxs_w.flatten() # (H*W,)
 

    def convertDepth(self, depths):
        """
        Convert depth img using ToF sensor model. Set all unknown depths to nan.
        Args:
            depths: depth img; array of shape (N, H*W)
        Returns:
            depths: depth img converted to ToF sensor array; array of shape (N, H*W)
        """
        depths = np.copy(depths)
        depths = depths.reshape(depths.shape[0], self.H, self.W) # (N, H, W)

        depths_out = np.full_like(depths, np.nan) # (N, H, W)
        depths_out[:, self.idxs_h, self.idxs_w] = depths[:, self.idxs_h, self.idxs_w]
        depths_out = depths_out.reshape(depths.shape[0], -1) # (N, H*W)

        return depths_out


class USSModel(SensorModel):
    def __init__(self, args, img_wh, num_imgs) -> None:
        SensorModel.__init__(self, args, img_wh)
        # define USS opening angle
        r = self.AoV2pixel(aov_sensor=args.uss.angle_of_view) / 2

        # create mask
        m1, m2 = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
        m1 = m1 - self.H/2 
        m2 = m2 - self.W/2
        self.mask = np.sqrt(m1**2 + m2**2) < r # (H, W)
        self.mask = self.mask.flatten() # (H*W,)  

        self.imgs_min_depth = torch.ones((num_imgs), dtype=torch.float32).to(self.args.device) * np.inf
        self.imgs_min_idx = torch.ones((num_imgs), dtype=torch.int32).to(self.args.device) * -1
        self.imgs_min_counts = torch.zeros((num_imgs), dtype=torch.int32).to(self.args.device)

    def convertDepth(self, depths:np.array):
        """
        Down sample depths from depth per pixel to depth per uss/img.
        Args:
            depths: depths per pixel; array of shape (N, H*W)
        Returns:
            depths_out: depths per uss; array of shape (N, H*W)
        """
        depths = np.copy(depths) # (N, H*W)

        d_min = np.nanmin(depths[:, self.mask], axis=1) # (N,)

        depths_out = np.full_like(depths, np.nan) # (N, H*W)
        depths_out[:, self.mask] = d_min[:,None]  

        # # get closest pixels inside mask and the corresponding indices
        # depths_m = depths[:, self.mask] # (N, M)
        # d_min = np.nanmin(depths_m, axis=1) # (N,)
        # d_idxs = np.where(depths_m==d_min[:,None]) # (N,), (N,)

        # depths_m_out = np.full_like(depths_m, np.nan) # (N, M)
        # depths_m_out[d_idxs[0], d_idxs[1]] = depths_m[d_idxs[0], d_idxs[1]]

        # depths_out = np.full_like(depths, np.nan) # (N, H*W)
        # depths_out[:, self.mask] = depths_m_out

        return depths_out
    

    def updateDepthMin(
            self, 
            results:dict, 
            data:dict,
    ):
        """
        Update the minimum depth of each image and the corresponding pixel index.
        Args:
            results: results of inference; dict
            data: data; dict
        Returns:
            imgs_depth_min: minimum depth per image; tensor of shape (num_imgs,)
            weights: weights for loss; tensor of shape (num_imgs,)
        """
        # mask data
        uss_mask = ~torch.isnan(data['depth']['USS']) # (N,)
        img_idxs = data['img_idxs'][uss_mask] # (n,)
        pix_idxs = data['pix_idxs'][uss_mask] # (n,)
        depths = results['depth'][uss_mask] # (n,)

        self.imgs_min_counts[img_idxs] += 1
        # weights = torch.exp(-self.imgs_min_counts/1000).to(self.args.device)
        weights = 1 - 1/(1 + self.imgs_min_counts/100)

        # # increase imgs_min_depth to avoid stagnation
        # self.imgs_min_depth[img_idxs] *= 1 + 1/(1 + self.imgs_min_counts[img_idxs])
        
        # determine minimum depth per image of batch
        min_depth_batch = torch.ones((len(self.imgs_min_depth), len(img_idxs)), dtype=torch.float).to(self.args.device) * np.inf # (num_imgs, n)
        min_depth_batch[img_idxs, np.arange(len(img_idxs))] = depths
        min_idx_batch = torch.argmin(min_depth_batch, dim=1) # (num_imgs,)
        min_idx_pix = pix_idxs[min_idx_batch] # (num_imgs,)
        min_depth_batch = min_depth_batch[torch.arange(len(min_idx_batch)), min_idx_batch] # (num_imgs,)

        # update minimum depth and minimum indices
        min_depth_temp = torch.where(
            condition=(min_idx_pix == self.imgs_min_idx),
            input=min_depth_batch,
            other=torch.minimum(self.imgs_min_depth, min_depth_batch)
        ) # (num_imgs,)
        self.imgs_min_idx = torch.where(
            condition=(min_idx_pix == self.imgs_min_idx),
            input=min_idx_pix,
            other=torch.where(
                condition=(self.imgs_min_depth <= min_depth_batch),
                input=self.imgs_min_idx,
                other=min_idx_pix
            )
        ) # (num_imgs,)
        self.imgs_min_depth = min_depth_temp     

        return self.imgs_min_depth.clone().detach(), weights



# class ComplexUSSModel(SensorModel):
#     def __init__(self, img_wh) -> None:
#         SensorModel.__init__(self, img_wh)

#         self.pool_size = 16
#         std = np.minimum(self.W, self.H) / 8
#         detection_prob_max = 0.2
#         detection_prob_min = 0.0
        
#         self.h = self.H // self.pool_size
#         self.w = self.W // self.pool_size
#         loc = np.array([self.h/2, self.w/2])
#         cov = np.array([[std**2, 0], [0, std**2]])
#         m1, m2 = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing='ij') # (h, w)
#         pos = np.stack((m1.flatten(), m2.flatten()), axis=1) # (h*w, 2)
#         self.gaussian = (1 / (2 * np.pi**2 * np.linalg.det(cov)**0.5)) \
#                         * np.exp(-0.5 * np.sum(((pos - loc) @ np.linalg.inv(cov)) * (pos - loc), axis=1)) # (h*w,)
        
#         self.gaussian = (self.gaussian - self.gaussian.min()) / (self.gaussian.max() - self.gaussian.min()) # normalize to [0,1]
#         self.gaussian = detection_prob_min + (detection_prob_max - detection_prob_min) * self.gaussian

#         self.rng = np.random.default_rng() # TODO: provide seed        

#     def convertDepth(self, depths:np.array, return_prob:bool=False):
#         """
#         Down sample depths from depth per pixel to depth per uss/img.
#         Closest pixel (c1) is chosen with probability of gaussian distribution: p(c1)
#         Second closest depth (c2) is chosen with probability: p(c2) = p(c2) * (1-p(c1))
#         Hence: p(ci) = sum(1-p(cj)) * p(ci) where the sum is over j = 1 ... i-1.
#         Args:
#             depths: depths per pixel; array of shape (N, H*W)
#         Returns:
#             depths_out: depths per uss; array of shape (N, h*w)
#             prob: probability of each depth; array of shape (N, h*w)
#         """
#         depths = np.copy(depths) # (N, H*W)
#         N = depths.shape[0]

#         depths = skimage.measure.block_reduce(depths.reshape(N, self.H, self.W), (1,self.pool_size,self.pool_size), np.min) # (N, h, w)
#         depths = depths.reshape(N, -1) # (N, h*w)

#         depths_out = np.zeros_like(depths)
#         probs = np.zeros_like(depths)
#         for i in range(N):
#             # get indices of sorted depths
#             sorted_idxs = np.argsort(depths[i])

#             # get probability of each depth
#             prob_sorted  = self.gaussian[sorted_idxs]
#             prob_sorted = np.cumprod((1-prob_sorted)) * prob_sorted / (1 - prob_sorted)
#             probs[i, sorted_idxs] = prob_sorted
#             probs[i, np.isnan(depths[i])] = 0.0
#             probs[i] = probs[i] / np.sum(probs[i])

#             # Choose random depth
#             rand_idx = self.rng.choice(depths.shape[1], size=1, p=probs[i])
#             depths_out[i,:] = depths[i, rand_idx]

#         if return_prob:
#             return depths_out, probs
#         return depths_out