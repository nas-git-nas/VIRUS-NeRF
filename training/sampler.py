import numpy as np
import torch

from args.args import Args


class Sampler():
    def __init__(
            self,
            args:Args,
            dataset_len:int,
            img_wh:tuple,
            seed:int,
            sensors_dict:dict=None,
            fct_getValidDepthMask:callable=None,
    ) -> None:
        """
        Args:
            args: hyper parameters; instance of Args class
            dataset_len: number of images in dataset; int
            img_wh: image width and height; tuple of int (2,)
            seed: random seed; int
            sensors_dict: dictionary of sensor models; dict
        """
        self.args = args
        self.dataset_len = dataset_len
        self.img_wh = img_wh
        self.sensors_dict = sensors_dict
        self.fct_getValidDepthMask = fct_getValidDepthMask

        self.rgn = np.random.default_rng(seed=seed)

        pixels = np.arange(self.img_wh[1]) - self.img_wh[1] / 2
        angles = pixels * self.args.rgbd.angle_of_view[1] / self.img_wh[1]
        weights = np.exp(- np.abs(angles) / 10)
        weights = np.repeat(weights.reshape(-1, 1), self.img_wh[0], axis=1)
        self.weights = (weights / np.sum(weights)).flatten()

        self.sample_count = {
            "nerf": torch.zeros(self.dataset_len, img_wh[0]*img_wh[1], dtype=torch.uint8, device=self.args.device),
            "occ": torch.zeros(self.dataset_len, img_wh[0]*img_wh[1], dtype=torch.uint8, device=self.args.device),
        }

    def __call__(
        self,
        batch_size:int,
        sampling_strategy:dict,
        origin:str,
    ):
        """
        Sample images and pixels/rays.
        Args:
            batch_size: number of samples; int
            sampling_strategy: sampling strategy used; dict
                'imgs': str
                    'all': sample from all images
                    'same': sample from the one image
                'rays': str
                    'random': sample random pixels
                    'ordered': sample equally distributed pixels with random offsets
                    'closest': sample random pixels but add some of the closest pixels
                    'weighted': sample random pixels with weights (more pixels in mid-height)
                    'valid_depth': sample random pixels with valid depth (not nan)
            origin: sampling origin; str
                    'nerf': sample for nerf
                    'occ': sample for occupancy grid
        Returns:
            img_idxs: indices of images to be used for training; tensor of int64 (batch_size,)
            pix_idxs: indices of pixels to be used for training; tensor of int64 (batch_size,)
        """
        img_idxs = self._imgIdxs(
            batch_size=batch_size,
            img_strategy=sampling_strategy["imgs"],
        )
        pix_idxs = self._pixIdxs(
            ray_strategy=sampling_strategy["rays"],
            img_idxs=img_idxs
        )
        count = self._count(
            img_idxs=img_idxs,
            pix_idxs=pix_idxs,
            origin=origin,
        )

        return img_idxs, pix_idxs, count
    
    def _imgIdxs(
            self,
            batch_size:int,
            img_strategy:str,
    ):
        """
        Sample image indices.
        Args:
            batch_size: number of samples; int
            img_strategy: image sampling strategy; str
        Returns:
            img_idxs: indices of images to be used for training; tensor of int64 (batch_size,)
        """
        if img_strategy == "all":
            return torch.randint(0, self.dataset_len, size=(batch_size,), device=self.args.device, dtype=torch.int32)
        
        if img_strategy == "same":
            idx = torch.randint(0, self.dataset_len, size=(1,), device=self.args.device, dtype=torch.int32)
            return idx * torch.ones(batch_size, device=self.args.device, dtype=torch.int32)               
        
        self.args.logger.error(f"image sampling strategy must be either 'all' or 'same'"
                + f" but is {self.args.training.sampling_strategy['imgs']}")

    def _pixIdxs(
            self,
            ray_strategy:str,
            img_idxs:torch.Tensor=None,
    ):
        """
        Sample pixel/ray indices.
        Args:
            ray_strategy: ray sampling strategy; str
            img_idxs: indices of images needed if ray sampling strategy is 'closest'; tensor of int64 (batch_size,)
        Returns:
            pix_idxs: indices of pixels to be used for training; tensor of int64 (batch_size,)
        """
        B = img_idxs.shape[0]
        WH = self.img_wh[0]*self.img_wh[1]

        if ray_strategy == "random":
            return torch.randint(0, WH, size=(B,), device=self.args.device, dtype=torch.int32)
        
        if ray_strategy == "ordered":
            step = WH / B
            pix_idxs = torch.linspace(0, WH-1-step, B, device=self.args.device)
            rand_offset = step * torch.rand(size=(B,), device=self.args.device)
            pix_idxs = torch.round(pix_idxs + rand_offset).to(torch.int64)
            pix_idxs = torch.clamp(pix_idxs, min=0, max=WH-1)
            return pix_idxs
        
        if ray_strategy == "entire_img":
            return torch.arange(0, WH, device=self.args.device, dtype=torch.int32)
        
        if ray_strategy == "closest":
            pix_idxs = torch.randint(0, WH, size=(B,), device=self.args.device, dtype=torch.int32)
            num_min_idxs = int(0.005 * B)
            pix_min_idxs = self.sensors_dict["USS"].imgs_min_idx
            pix_idxs[:num_min_idxs] = pix_min_idxs[img_idxs[:num_min_idxs]]
            return pix_idxs
        
        if ray_strategy == "weighted":
            pix_idxs = self.rgn.choice(WH, size=(B,), p=self.weights)
            return torch.from_numpy(pix_idxs).to(torch.int64)
        
        if ray_strategy == "valid_depth":
            val_idxs_dict = self.fct_getValidDepthMask(img_idxs) # dict of sensor: valid depth; bool tensor (B, H*W)

            # random ints used to create B random permutation of H*W intergers
            rand_ints = torch.randint(0, WH, (B, WH), device=self.args.device, dtype=torch.int32) # (B, H*W)

            # replace random ints with -n where depth is invalid in order to not sample them
            # n is the number of sensor for which the depth is invalid
            for val_idxs in val_idxs_dict.values():
                rand_ints = torch.where(
                    condition=val_idxs, 
                    input=rand_ints, 
                    other=torch.minimum(rand_ints-1, -torch.ones_like(rand_ints, device=self.args.device, dtype=torch.int32))
                ) # (B, H*W)

            # create random permutation for every row where invalid pixels are at the beginning   
            perm_idxs = torch.argsort(rand_ints, dim=1) # (B, H*W)
            return perm_idxs[:,-1] # (B), random pixel indices with valid depth (except entire row is invalid)
        
        if ray_strategy == "uss_tof_split":
            tof_num_samples = int(B/2)
            tof_mask = torch.tensor(self.sensors_dict["ToF"].mask, device=self.args.device, dtype=torch.bool)
            tof_mask_idxs = torch.where(tof_mask)[0]
            tof_rand_ints = torch.randint(0, tof_mask_idxs.shape[0], (tof_num_samples,), device=self.args.device, dtype=torch.int32)
            tof_img_idxs = tof_mask_idxs[tof_rand_ints]

            uss_num_samples = B - tof_num_samples
            uss_maks = torch.tensor(self.sensors_dict["USS"].mask, device=self.args.device, dtype=torch.bool)
            uss_mask_idxs = torch.where(uss_maks)[0]
            uss_rand_ints = torch.randint(0, uss_mask_idxs.shape[0], (uss_num_samples,), device=self.args.device, dtype=torch.int32)
            uss_img_idxs = uss_mask_idxs[uss_rand_ints]

            return torch.cat((uss_img_idxs, tof_img_idxs), dim=0)
        
        print(f"ERROR: sampler._pixIdxs: pixel sampling strategy must be either 'random', 'ordered', 'closest' or weighted"
              + f" but is {self.args.training.sampling_strategy['rays']}")
        
    def _count(
        self,
        img_idxs:torch.Tensor,
        pix_idxs:torch.Tensor,
        origin:str,
    ):
        """
        Count how often a pixel is sampled for each origin.
        Args:
            img_idxs: indices of images used for training; tensor of int64 (batch_size,)
            pix_idxs: indices of pixels used for training; tensor of int64 (batch_size,)
            origin: sampling origin; str
                    'nerf': sample for nerf
                    'occ': sample for occupancy grid
        Returns:
            count: number of times a pixel is sampled for origin; tensor of uint8 (batch_size,)
        """
        if origin == "nerf":
            count_is_max = self.sample_count["nerf"][img_idxs, pix_idxs] == 255
            if torch.any(count_is_max):
                self.args.logger.warning(f"overflows from 255->0: limit count to 255")
            self.sample_count["nerf"][img_idxs, pix_idxs] = torch.where(
                condition=count_is_max,
                input=self.sample_count["nerf"][img_idxs, pix_idxs],
                other=self.sample_count["nerf"][img_idxs, pix_idxs] + 1,
            )
            return self.sample_count["nerf"][img_idxs, pix_idxs]
        
        if origin == "occ":
            count_is_max = self.sample_count["occ"][img_idxs, pix_idxs] == 255
            if torch.any(count_is_max):
                self.args.logger.warning(f"sampler._count: overflows from 255->0: limit count to 255")
            self.sample_count["occ"][img_idxs, pix_idxs] = torch.where(
                condition=count_is_max,
                input=self.sample_count["occ"][img_idxs, pix_idxs],
                other=self.sample_count["occ"][img_idxs, pix_idxs] + 1,
            )
            return self.sample_count["occ"][img_idxs, pix_idxs]

        self.args.logger.error(f"origin must be either 'nerf' or 'occ'")