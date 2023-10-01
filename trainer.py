import glob
import os
import time
import tqdm
import random
import warnings

import torch
import imageio
import numpy as np
import taichi as ti
from einops import rearrange
import torch.nn.functional as F
from abc import abstractmethod

from gui import NGPGUI
from opt import get_opts
from args import Args
from datasets import dataset_dict
from datasets.ray_utils import get_rays

from modules.networks import NGP
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model

from torchmetrics import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
)



warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, dataset, model_config) -> None:
        # hyper parameters
        self.hparams = get_opts()
        
        # variable arguments
        self.args = Args(hparams=self.hparams)

        # TODO: add as hparams
        if self.args.device == torch.device("cuda"):
            root_dir =  '/media/scratch1/schmin/data/robot_at_home'
        else:
            root_dir =  '../RobotAtHome2/data'

        # set seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        self.taichi_init(self.hparams)

        # datasets       
        self.train_dataset = dataset(
            root_dir=root_dir,
            split="train",
            downsample=self.hparams.downsample,
        ).to(self.args.device)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(
            root_dir=root_dir,
            split='test',
            downsample=self.hparams.downsample,
        ).to(self.args.device)

        # model
        self.model = NGP(**model_config).to(self.args.device)

        # load checkpoint if ckpt path is provided
        if self.hparams.ckpt_path:
            self.__loadCheckpoint(ckpt_path=self.hparams.ckpt_path)

        self.model.mark_invisible_cells(
            self.train_dataset.K,
            self.train_dataset.poses, 
            self.train_dataset.img_wh,
        )

        # use large scaler, the default scaler is 2**16 
        # TODO: investigate why the gradient is small
        if self.hparams.half_opt:
            scaler = 2**16
        else:
            scaler = 2**19
        self.grad_scaler = torch.cuda.amp.GradScaler(scaler)

        # optimizer
        self.lr = 1e-2 # TODO: add as hparams
        try:
            import apex
            self.optimizer = apex.optimizers.FusedAdam(
                self.model.parameters(), 
                lr=self.lr, 
                eps=1e-15,
            )
        except ImportError:
            print("Failed to import apex FusedAdam, use torch Adam instead.")
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                self.lr, 
                eps=1e-15,
            )

        # scheduler
        self.hparams.max_steps = 3000 # TODO: add as hparams
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.hparams.max_steps,
            eta_min=self.lr/30,
    )

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def saveModel(self):
        """
        Save model
        """
        print(f"Saving model to {self.args.val_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.args.val_dir, 'model.pth'),
        )

    def loadCheckpoint(self, ckpt_path:str):
        """
        Load checkpoint
        Args:
            ckpt_path: path to checkpoint; str
        """
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print("Load checkpoint from %s" % ckpt_path)

    def taichi_init(self, args):
        taichi_init_args = {"arch": ti.cuda,}
        if args.half_opt:
            taichi_init_args["half2_vectorization"] = True

        ti.init(**taichi_init_args)

    def lossFunc(self, results, data, depth_loss_w=1.0):
        """
        Loss function for training
        Args:
            results: dict of rendered images
                'opacity': sum(transmittance*alpha); array of shape: (N,)
                'depth': sum(transmittance*alpha*t__i); array of shape: (N,)
                'rgb': sum(transmittance*alpha*rgb_i); array of shape: (N, 3)
                'total_samples': total samples for all rays; int
                where   transmittance = exp( -sum(sigma_i * delta_i) )
                        alpha = 1 - exp(-sigma_i * delta_i)
                        delta_i = t_i+1 - t_i
            data: dict of ground truth images
                'img_idxs': image indices; array of shape (N,) or (1,) if same image
                'pix_idxs': pixel indices; array of shape (N,)
                'pose': poses; array of shape (N, 3, 4)
                'direction': directions; array of shape (N, 3)
                'rgb': pixel colours; array of shape (N, 3)
                'depth': pixel depths; array of shape (N,)
            depth_loss_w: weight of depth loss; float
        Returns:
            loss: loss value; float
        """
        colour_loss = F.mse_loss(results['rgb'], data['rgb'])

        val_idxs = ~torch.isnan(data['depth'])
        depth_loss = F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])
        return colour_loss + depth_loss_w * depth_loss
    


