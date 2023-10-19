import numpy as np
import torch

from args.args import Args


class Sampler():
    def __init__(
            self,
            args:Args,
            dataset_len:int,
            img_wh:tuple,
            sensors_dict:dict=None,
    ) -> None:
        """
        Args:
            args: hyper parameters; instance of Args class
            dataset_len: number of images in dataset; int
            img_wh: image width and height; tuple of int (2,)
            sensors_dict: dictionary of sensor models; dict
        """
        self.args = args
        self.dataset_len = dataset_len
        self.img_wh = img_wh
        self.sensors_dict = sensors_dict

    def __call__(
            self,
    ):
        """
        Sample images and pixels/rays.
        Returns:
            img_idxs: indices of images to be used for training; tensor of int64 (batch_size,)
            pix_idxs: indices of pixels to be used for training; tensor of int64 (batch_size,)
        """
        img_idxs = self._imgIdxs()
        pix_idxs = self._pixIdxs(
            img_idxs=img_idxs
        )

        return img_idxs, pix_idxs
    
    def _imgIdxs(
            self,
    ):
        """
        Sample image indices.
        Returns:
            img_idxs: indices of images to be used for training; tensor of int64 (batch_size,)
        """
        if self.args.training.sampling_strategy["imgs"] == "all":
            return torch.randint(0, self.dataset_len, size=(self.args.training.batch_size,), device=self.args.device)
        
        if self.args.training.sampling_strategy["imgs"] == "same":
            idx = torch.randint(0, self.dataset_len, size=(1,), device=self.args.device)
            return idx * torch.ones(self.args.training.batch_size, dtype=torch.int64, device=self.args.device)               
        
        print(f"ERROR: sampler._imgIdxs: image sampling strategy must be either 'all' or 'same'"
              + f" but is {self.args.training.sampling_strategy['imgs']}")

    def _pixIdxs(
            self,
            img_idxs:torch.Tensor=None,
    ):
        """
        Sample pixel/ray indices.
        Args:
            img_idxs: indices of images needed if ray sampling strategy is 'closest'; tensor of int64 (batch_size,)
        Returns:
            pix_idxs: indices of pixels to be used for training; tensor of int64 (batch_size,)
        """
        if self.args.training.sampling_strategy["rays"] == "random":
            return torch.randint(0, self.img_wh[0]*self.img_wh[1], size=(self.args.training.batch_size,), device=self.args.device)
        
        if self.args.training.sampling_strategy["rays"] == "ordered":
            step = self.img_wh[0]*self.img_wh[1] / self.args.training.batch_size
            pix_idxs = torch.linspace(0, self.img_wh[0]*self.img_wh[1]-1-step, self.args.training.batch_size, device=self.args.device)
            rand_offset = step * torch.rand(size=(self.args.training.batch_size,), device=self.args.device)
            pix_idxs = torch.round(pix_idxs + rand_offset).to(torch.int64)
            pix_idxs = torch.clamp(pix_idxs, min=0, max=self.img_wh[0]*self.img_wh[1]-1)
            return pix_idxs
        
        if self.args.training.sampling_strategy["rays"] == "closest":
            pix_idxs = torch.randint(0, self.img_wh[0]*self.img_wh[1], size=(self.args.training.batch_size,), device=self.args.device)
            num_min_idxs = int(0.005 * self.args.training.batch_size)
            pix_min_idxs = self.sensors_dict["USS"].imgs_min_idx
            pix_idxs[:num_min_idxs] = pix_min_idxs[img_idxs[:num_min_idxs]]
            return pix_idxs
        
        print(f"ERROR: sampler._pixIdxs: pixel sampling strategy must be either 'random' or 'ordered'"
              + f" but is {self.args.training.sampling_strategy['rays']}")