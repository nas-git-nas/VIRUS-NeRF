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
from args.args import Args
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from datasets.robot_at_home import RobotAtHomeDataset

from modules.networks import NGP
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model

from torchmetrics import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
)



warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, hparams_file) -> None:
        
        # get hyper-parameters and other variables
        self.args = Args(file_name=hparams_file)

        # set seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        self.taichi_init()

        # datasets   
        if self.args.dataset.name == 'robot_at_home':
            dataset = RobotAtHomeDataset    
        
        self.train_dataset = dataset(
            args = self.args,
            split="train",
        ).to(self.args.device)
        self.train_dataset.batch_size = self.args.training.batch_size
        self.train_dataset.ray_sampling_strategy = self.args.training.ray_sampling_strategy

        self.test_dataset = dataset(
            args = self.args,
            split='test',
        ).to(self.args.device)

        # model
        model_config = {
            'scale': self.args.model.scale,
            'pos_encoder_type': self.args.model.encoder_type,
            'max_res': self.args.occ_grid.max_res, 
            'half_opt': False, # TODO: args
        }
        self.model = NGP(**model_config).to(self.args.device)

        # load checkpoint if ckpt path is provided
        if self.args.model.ckpt_path:
            self.loadCheckpoint(ckpt_path=self.hpaargs.modelrams.ckpt_path)

        self.model.mark_invisible_cells(
            self.train_dataset.K,
            self.train_dataset.poses, 
            self.train_dataset.img_wh,
        )

        # # use large scaler, the default scaler is 2**16 
        # # TODO: investigate why the gradient is small
        # if self.hparams.half_opt:
        #     scaler = 2**16
        # else:
        #     scaler = 2**19
        scaler = 2**19
        self.grad_scaler = torch.cuda.amp.GradScaler(scaler)

        # optimizer
        try:
            import apex
            self.optimizer = apex.optimizers.FusedAdam(
                self.model.parameters(), 
                lr=self.args.training.lr, 
                eps=1e-15,
            )
        except ImportError:
            print("Failed to import apex FusedAdam, use torch Adam instead.")
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                self.args.training.lr, 
                eps=1e-15,
            )

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.training.max_steps,
            eta_min=self.args.training.lr/30,
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
        print(f"Saving model to {self.args.save_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.args.save_dir, 'model.pth'),
        )
        self.args.saveJson()

    def loadCheckpoint(self, ckpt_path:str):
        """
        Load checkpoint
        Args:
            ckpt_path: path to checkpoint; str
        """
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print("Load checkpoint from %s" % ckpt_path)

    def taichi_init(self):
        taichi_init_args = {"arch": ti.cuda,}
        # if hparams.half_opt:
        #     taichi_init_args["half2_vectorization"] = True

        ti.init(**taichi_init_args)

    def lossFunc(self, results, data):
        """
        Loss function for training
        Args:
            results: dict of rendered images
                'opacity': sum(transmittance*alpha); array of shape: (N,)
                'depth': sum(transmittance*alpha*t_i); array of shape: (N,)
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
        Returns:
            total_loss: loss value; float
            colour_loss: colour loss value; float
            depth_loss: depth loss value; float
        """
        colour_loss = F.mse_loss(results['rgb'], data['rgb'])

        # val_idxs = ~torch.isnan(data['depth'])
        # depth_loss = self.args.training.depth_loss_w * F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])
        # if torch.all(torch.isnan(depth_loss)):
        #     print("WARNING: trainer:lossFunc: depth_loss is nan, set to 0.")
        #     depth_loss = 0

        depth_loss = 0.0
        if self.args.dataset.name == 'robot_at_home':
            if self.args.rh.sensor_model == 'USS':
                uss_mask = ~torch.isnan(data['depth'])
                too_close = results['depth'] < data['depth']
                if torch.any(too_close & uss_mask):
                    depth_loss = F.mse_loss(results['depth'][too_close & uss_mask], data['depth'][too_close & uss_mask])
            if self.args.rh.sensor_model == 'ToF':
                val_idxs = ~torch.isnan(data['depth'])
                depth_loss = F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])
        depth_loss *= self.args.training.depth_loss_w
        
        total_loss = colour_loss + depth_loss
        return total_loss, colour_loss, depth_loss
    


