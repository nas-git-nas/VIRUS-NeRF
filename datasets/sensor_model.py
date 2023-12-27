import numpy as np
import torch
import matplotlib.pyplot as plt
from abc import abstractmethod
import skimage.measure
from typing import TypedDict

from args.args import Args
from helpers.data_fcts import sensorName2ID, sensorID2Name


class SensorModel():
    def __init__(
        self, 
        args:Args, 
        img_wh:tuple
    ) -> None:
        self.args = args
        self.W = img_wh[0]
        self.H = img_wh[1]
        
    @abstractmethod
    def convertDepth(self, depths):
        pass

    def pos2idx(
        self, 
        pos_h:np.array, 
        pos_w:np.array,
    ):
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

    def AoV2pixel(
        self, 
        aov_sensor:list
    ):
        """
        Convert the angle of view to width in pixels
        Args:
            aov_sensor: angle of view of sensor in width and hight; list
        Returns:
            num_pixels: width in pixels; int
        """
        img_wh = np.array([self.W, self.H])
        aov_sensor = np.array(aov_sensor)
        aov_camera = self.args.rgbd.angle_of_view

        num_pixels = img_wh * aov_sensor / aov_camera
        return np.round(num_pixels).astype(int)
    

class RGBDModel(SensorModel):
    def __init__(
        self, 
        args:Args, 
        img_wh:tuple
    ) -> None:
        """
        Sensor model for Time of Flight (ToF) sensor.
        Args:
            img_wh: image width and height, tuple of int
        """
        SensorModel.__init__(self, args, img_wh)     

    def convertDepth(
        self, 
        depths:np.array,
        format:str="img",
    ):
        """
        Convert depth img using ToF sensor model. Set all unknown depths to nan.
        Args:
            depths: depth img; array of shape (N, H*W)
            format: not used
        Returns:
            depths: depth img converted to ToF sensor array; array of shape (N, H*W)
        """
        return np.copy(depths)


class ToFModel(SensorModel):
    def __init__(
        self, 
        args:Args, 
        img_wh:tuple
    ) -> None:
        """
        Sensor model for Time of Flight (ToF) sensor.
        Args:
            img_wh: image width and height, tuple of int
        """
        SensorModel.__init__(
            self, 
            args=args, 
            img_wh=img_wh,
        )    

        self.mask = self._createMask() # (H*W,)
        self.error_mask = self._createErrorMask(
            mask=self.mask.detach().clone().cpu().numpy(),
        ) # (H*W,)
        

    def convertDepth(
            self, 
            depths:np.array,
            format:str="img",
        ):
        """
        Convert depth img using ToF sensor model. Set all unknown depths to nan.
        Args:
            depths: depth img
            format: depths format; str
                    "img": depth per camera pixel; depths array of shape (N, H*W)
                    "sensor": depth per ToF pixel; depths array of shape (N, 8*8)
        Returns:
            depths: depth img converted to ToF sensor array; array of shape (N, H*W)
        """
        depths = np.copy(depths) # (N, H*W)
        depths_out = np.full((depths.shape[0], self.H*self.W), np.nan) # (N, H*W)
        fov_mask = self.mask.detach().clone().cpu().numpy() # (H*W,)
        error_mask = self.error_mask.detach().clone().cpu().numpy() # (H*W,)

        if format == "img":
            depths_out[:, fov_mask] = depths[:,error_mask] 
        elif format == "sensor":
            depths_out[:, fov_mask] = depths
        else:
            self.args.logger.error(f"Unknown depth format: {format}")

        if (self.args.tof.sensor_random_error == 0.0) or (self.args.tof.sensor_random_error is None):
            return depths_out
        
        # add random error to depths
        self.args.logger.info(f"Add random error to ToF depths: {self.args.tof.sensor_random_error}°")
        valid_depths = ~np.isnan(depths_out) # (N, H*W)
        rand_error = np.random.normal(loc=0.0, scale=self.args.tof.sensor_random_error, size=depths_out.shape) # (N, H*W)
        depths_out[valid_depths] += rand_error[valid_depths]
        return depths_out
    
    def _createMask(
        self,
    ):
        """
        Create mask for ToF sensor.
        Returns:
            mask: mask for ToF sensor; tensor of shape (H*W,)
        """
        # calculate indices of ToF sensor array
        pix_wh = self.AoV2pixel(aov_sensor=self.args.tof.angle_of_view)
        idxs_w = np.linspace(0, pix_wh[0], self.args.tof.matrix[0], dtype=float)
        idxs_h = np.linspace(0, pix_wh[1], self.args.tof.matrix[1], dtype=float)

        # ajust indices to quadratic shape
        idxs_w = idxs_w + (self.W - pix_wh[0])/2
        idxs_h = idxs_h + (self.H - pix_wh[1])/2

        # convert indices to ints
        idxs_h, idxs_w = self.pos2idx(idxs_h, idxs_w) # (H,), (W,)     

        # create meshgrid of indices
        idxs_h, idxs_w = np.meshgrid(idxs_h, idxs_w, indexing='ij') # (H, W)
        self.idxs_h = idxs_h.flatten() # (H*W,)
        self.idxs_w = idxs_w.flatten() # (H*W,)

        # create mask
        mask = np.zeros((self.H, self.W), dtype=bool) # (H, W)
        mask[idxs_h, idxs_w] = True
        mask = torch.tensor(mask.flatten(), dtype=torch.bool).to(self.args.device) # (H*W,)
        return mask # (H*W,)
    
    def _createErrorMask(
        self,
        mask:torch.Tensor,
    ):
        """
        Create error mask for ToF sensor. If the calibration error is equal to 0.0, 
        the error mask is equal to the mask. Otherwise, the error mask is a shifted
        in a random direction by the calibration error. In this case, the ToF-depth is
        evaluated by using the error mask and assigned to the pixel in the mask.
        Args:
            error_mask: error mask for ToF sensor; tensor of shape (H*W,)
        """
        mask = np.copy(mask) # (H*W,)
        if self.args.tof.sensor_calibration_error == 0.0:
            return torch.tensor(mask, dtype=torch.bool).to(self.args.device)

        # determine error in degrees
        direction = 0.0
        error = self.args.tof.sensor_calibration_error * np.array([np.cos(direction), np.sin(direction)]).flatten()

        # convert error to pixels
        error[0] = self.H * error[0] / self.args.rgbd.angle_of_view[0]
        error[1] = self.W * error[1] / self.args.rgbd.angle_of_view[1]
        error = np.round(error).astype(int)

        # convert error to mask indices
        mask = mask.reshape(self.H, self.W)
        idxs = np.argwhere(mask)
        idxs[:,0] = np.clip(idxs[:,0] + error[0], 0, self.H-1)
        idxs[:,1] = np.clip(idxs[:,1] + error[1], 0, self.W-1)

        # apply error to mask
        error_mask = np.zeros((self.H, self.W), dtype=bool)
        error_mask[idxs[:,0], idxs[:,1]] = True
        error_mask = torch.tensor(error_mask.flatten(), dtype=torch.bool).to(self.args.device)
        return error_mask # (H*W,)


class USSModel(SensorModel):
    def __init__(
        self, 
        args:Args,
        img_wh:tuple,
        sensor_ids:np.ndarray,
    ) -> None:
        """
        USS sensor model.
        Args:
            args: arguments; Args
            img_wh: image width and height, tuple of int
            sensor_ids: sensor identities for every image; array of ints (N,)
        """
        SensorModel.__init__(
            self, 
            args=args, 
            img_wh=img_wh,
        )
        
        self.mask = self._createMask() # (H*W,)

        self.num_imgs = sensor_ids.shape[0]

        self.imgs_stats = {}
        for id in np.unique(sensor_ids):
            name = sensorID2Name(
                sensor_id=id, 
                sensor_type="USS",
                dataset=self.args.dataset.name,
            )

            img_idxs = np.where(sensor_ids == id)[0]

            self.imgs_stats[name] = {
                "id": id,
                "img_idxs": torch.tensor(img_idxs, dtype=torch.int32).to(self.args.device),
                "pix_idxs": torch.randint(0, self.W*self.H, size=(img_idxs.shape[0],), device=self.args.device, dtype=torch.int32),
                "depths": np.inf * torch.ones((img_idxs.shape[0]), dtype=torch.float32).to(self.args.device),
                "counts": torch.zeros((img_idxs.shape[0]), dtype=torch.int32).to(self.args.device),
            }

    def getStatsForBatch( 
        self,
        batch_img_idxs:torch.Tensor,
    ):
        """
        Get statistics for batch.
        Args:
            batch_img_idxs: image indices; tensor of shape (N_batch,)
        Returns:
            batch_pix_idxs: pixel indices; tensor of shape (N_batch,)
            batch_depths: minimum depth per batch; tensor of shape (N_batch,)
            batch_counts: update counts per batch; tensor of shape (N_batch,)
        """
        # get stats for all images
        imgs_pix_idxs = -1 * torch.ones((self.num_imgs), dtype=torch.int32).to(self.args.device) # (N_imgs,)
        imgs_depths = -1 * torch.ones((self.num_imgs), dtype=torch.float32).to(self.args.device) # (N_imgs,)
        imgs_counts = -1 * torch.ones((self.num_imgs), dtype=torch.int32).to(self.args.device) # (N_imgs,)
        for stats in self.imgs_stats.values():
            imgs_pix_idxs[stats["img_idxs"]] = stats["pix_idxs"] # (N_imgs,)
            imgs_depths[stats["img_idxs"]] = stats["depths"] # (N_imgs,)
            imgs_counts[stats["img_idxs"]] = stats["counts"] # (N_imgs,)

        # check if all minimum depths have been updated
        if self.args.model.debug_mode:
            if torch.any(imgs_depths < 0):
                self.args.logger.error(f"USSModel.updateDepthMin: imgs_depths < 0")
        
        # convert minimum pixels, depths and counts to batch
        batch_pix_idxs = imgs_pix_idxs[batch_img_idxs].clone().detach() # (N_batch,)
        batch_depths = imgs_depths[batch_img_idxs].clone().detach() # (N_batch,)
        batch_counts = imgs_counts[batch_img_idxs].clone().detach() # (N_batch,)
        return batch_pix_idxs, batch_depths, batch_counts

    def convertDepth(
        self, 
        depths:np.array,
        format:str="img",
    ):
        """
        Convert depth img using ToF sensor model. Set all unknown depths to nan.
        Args:
            depths: depth img
            format: depths format; str
                    "img": depth per camera pixel; depths array of shape (N, H*W)
                    "sensor": depth per ToF pixel; depths array of shape (N,)
        Returns:
            depths_out: depth img converted to ToF sensor array; array of shape (N, H*W)
        """
        depths = np.copy(depths) # (N, H*W) or (N,)
        depths_out = np.full((depths.shape[0], self.W*self.H), np.nan) # (N, H*W)
        fov_mask = self.mask.detach().clone().cpu().numpy() # (H*W,)

        if format == "img":
            d_min = np.nanmin(depths[:, fov_mask], axis=1) # (N,)
        elif format == "sensor":
            d_min = depths # (N,)
        else:
            self.args.logger.error(f"Unknown depth format: {format}")

        depths_out[:, fov_mask] = d_min[:,None] # (N, H*W)
        return depths_out   

    def updateStats( 
        self,
        depths:torch.Tensor,
        data:dict,
    ):
        """
        Update the minimum depth of each image and the corresponding pixel index.
        Args:
            depths: depth of forward pass; tensor of shape (N_batch,)
            data: data; dict
                    'img_idxs': image indices; tensor of shape (N_batch,)
                    'pix_idxs': pixel indices; tensor of shape (N_batch,)
                    'sensor_ids': sensor ids; tensor of shape (N_batch,)
        Returns:
            batch_min_depths: minimum depth per batch; tensor of shape (N_batch,)
            batch_min_counts: updated counts per batch; tensor of shape (N_batch,)
        """
        for stats in self.imgs_stats.values():
            stats = self._updateSensorStats(
                stats=stats,
                batch_depths=depths,
                data=data,
            )

        batch_min_pix_idxs, batch_min_depths, batch_min_counts = self.getStatsForBatch(
            batch_img_idxs=data["img_idxs"],
        )
        return batch_min_depths, batch_min_counts

    def _updateSensorStats(
        self,
        stats:dict,
        batch_depths:torch.Tensor,
        data:dict,
    ):
        """
        Update the minimum depth of each image and the corresponding pixel index.
        Args:
            stats: sensor stats; dict
                    'id': sensor id; int
                    'img_idxs': image indices; tensor of shape (N_sensor,)
                    'pix_idxs': pixel indices; tensor of shape (N_sensor,)
                    'depths': minimum depth; tensor of shape (N_sensor,)
                    'counts': update counts; tensor of shape (N_sensor,)
            batch_depths: depth of forward pass; tensor of shape (N_batch,)
            data: data; dict
                    'img_idxs': image indices; tensor of shape (N_batch,)
                    'pix_idxs': pixel indices; tensor of shape (N_batch,)
                    'sensor_ids': sensor ids; tensor of shape (N_batch,)
        Returns:
            stats: updated sensor stats; dict
        """
        sensor_id = stats["id"]
        sensor_min_img_idxs = stats["img_idxs"] # (N_sensor,)
        sensor_min_pix_idxs = stats["pix_idxs"] # (N_sensor,)
        sensor_min_depths = stats["depths"] # (N_sensor,)
        sensor_min_counts = stats["counts"] # (N_sensor,)

        batch_img_idxs = data["img_idxs"] # (N_batch,)
        batch_pix_idxs = data["pix_idxs"] # (N_batch,)
        batch_ids = data["sensor_ids"] # (N_batch,)

        # use only samples in field-of-view and of particular sensor to update stats
        fov_mask = torch.tensor(self.mask[batch_pix_idxs], dtype=torch.bool).to(self.args.device) # (N_batch,)
        sensor_mask = (batch_ids == sensor_id) # (N_batch,)
        mask = fov_mask & sensor_mask # (N_batch,)

        batch_img_idxs = batch_img_idxs[mask] # (n_batch,)
        batch_pix_idxs = batch_pix_idxs[mask] # (n_batch,)
        batch_depths = batch_depths[mask] # (n_batch,)

        # deterimne minimum depth contained in batch for every image
        batch_min_depths = np.inf * torch.ones((self.num_imgs, len(batch_img_idxs)), dtype=torch.float).to(self.args.device) # (N_imgs, n_batch)
        batch_min_depths[batch_img_idxs, np.arange(len(batch_img_idxs))] = batch_depths # (N_imgs, n_batch)
        batch_min_depths, min_idxs = torch.min(batch_min_depths, dim=1) # (N_imgs,), (N_imgs,)
        batch_min_pix_idxs = batch_pix_idxs[min_idxs] # (N_imgs,)

        # deterimne minimum depth contained in batch for particular sensor
        batch_min_depths = batch_min_depths[sensor_min_img_idxs] # (N_sensor,)
        batch_min_pix_idxs = batch_min_pix_idxs[sensor_min_img_idxs] # (N_sensor,)

        # update minimum depth and minimum pixel indices
        sensor_min_depths = torch.where(
            condition=(batch_min_pix_idxs == sensor_min_pix_idxs),
            input=batch_min_depths,
            other=torch.minimum(batch_min_depths, sensor_min_depths)
        ) # (N_sensor,)
        sensor_min_pix_idxs = torch.where(
            condition=(batch_min_depths <= sensor_min_depths),
            input=batch_min_depths,
            other=sensor_min_depths,
        ) # (N_sensor,)

        # update minimum counts
        batch_counts = torch.zeros((self.num_imgs), dtype=torch.int32).to(self.args.device) # (N_imgs,)
        batch_counts[batch_img_idxs] = 1 # (N_imgs,)
        batch_counts = batch_counts[sensor_min_img_idxs] # (N_sensor,)
        sensor_min_counts = sensor_min_counts + batch_counts # (N_sensor,)

        # update stats
        stats["img_idxs"] = sensor_min_img_idxs.to(dtype=torch.int32)
        stats["pix_idxs"] = sensor_min_pix_idxs.to(dtype=torch.int32)
        stats["depths"] = sensor_min_depths.to(dtype=torch.float32)
        stats["counts"] = sensor_min_counts.to(dtype=torch.int32)
        return stats

    def _createMask(
        self,
    ) -> torch.Tensor:
        """
        Create mask for ToF sensor.
        Returns:
            mask: mask for ToF sensor; tensor of shape (H*W,)
        """
        # define USS opening angle
        pix_wh = self.AoV2pixel(
            aov_sensor=self.args.uss.angle_of_view
        ) # (2,)
        pix_wh = (pix_wh/2.0).astype(np.int32) # convert diameter to radius

        # create mask
        m1, m2 = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
        m1 = m1 - self.H/2 
        m2 = m2 - self.W/2
        mask = (m1/pix_wh[1])**2 + (m2/pix_wh[0])**2 < 1 # (H, W), ellipse
        mask = torch.tensor(mask.flatten(), dtype=torch.bool).to(self.args.device) # (H*W,)
        return mask # (H*W,)  
    




        # # mask data
        # uss_mask = ~torch.isnan(data_depth_uss) # (N,)
        # img_idxs_n = img_idxs[uss_mask] # (n,)
        # pix_idxs_n = pix_idxs[uss_mask] # (n,)
        # depths = results_depth[uss_mask] # (n,)

        # imgs_min_counts[img_idxs_n] += 1
        
        # # determine minimum depth per image of batch
        # min_depth_batch = torch.ones((len(imgs_min_depth), len(img_idxs_n)), dtype=torch.float).to(self.args.device) * np.inf # (num_imgs, n)
        # min_depth_batch[img_idxs_n, np.arange(len(img_idxs_n))] = depths
        # min_idx_batch = torch.argmin(min_depth_batch, dim=1) # (num_imgs,)
        # min_idx_pix = pix_idxs_n[min_idx_batch] # (num_imgs,)
        # min_depth_batch = min_depth_batch[torch.arange(len(min_idx_batch)), min_idx_batch] # (num_imgs,)

        # # update minimum depth and minimum indices
        # min_depth_temp = torch.where(
        #     condition=(min_idx_pix == imgs_min_idx),
        #     input=min_depth_batch,
        #     other=torch.minimum(imgs_min_depth, min_depth_batch)
        # ) # (num_imgs,)
        # imgs_min_idx = torch.where(
        #     condition=(min_idx_pix == imgs_min_idx),
        #     input=min_idx_pix,
        #     other=torch.where(
        #         condition=(imgs_min_depth <= min_depth_batch),
        #         input=imgs_min_idx,
        #         other=min_idx_pix
        #     )
        # ) # (num_imgs,)
        # imgs_min_depth = min_depth_temp # (N_img,)

        # # return minimum depth and weights of batch
        # depths_min = imgs_min_depth[img_idxs].clone().detach() # (N,)
        # weights = weights[img_idxs].clone().detach() # (N,)

        # # update stats
        # stats["min_depth"] = imgs_min_depth
        # stats["min_idx"] = imgs_min_idx
        # stats["min_counts"] = imgs_min_counts
        # return stats

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