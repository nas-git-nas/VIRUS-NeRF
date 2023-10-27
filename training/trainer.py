import glob
import os
import time
import tqdm
import random
import warnings

import torch
import imageio
import numpy as np
import pandas as pd
import taichi as ti
from einops import rearrange
import torch.nn.functional as F
from abc import abstractmethod

# from gui import NGPGUI
# from opt import get_opts
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
            'rh_scene': self.train_dataset.scene,
            'args': self.args,
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

    @abstractmethod
    def lossFunc(self):
        pass

    def saveModel(self):
        """
        Save model, args and logs
        """
        print(f"Saving model to {self.args.save_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.args.save_dir, 'model.pth'),
        )
        self.args.saveJson()

        # remove empty logs
        del_keys = []
        for key in self.logs.keys():
            if len(self.logs[key]) == 0:
                del_keys.append(key)
        for key in del_keys:
            del self.logs[key]

        # save logs
        logs_df = pd.DataFrame(self.logs)
        logs_df.to_csv(os.path.join(self.args.save_dir, 'logs.csv'), index=False)

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


    


