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

from alive_progress import alive_bar
from contextlib import nullcontext

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

from datasets import dataset_dict
from datasets.ray_utils import get_rays

from modules.networks import NGP
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model
from helpers.geometric_fcts import findNearestNeighbour
from helpers.data_fcts import linInterpolateArray, convolveIgnorNans, dataConverged
from training.metrics_rh import MetricsRH

from modules.occupancy_grid import OccupancyGrid

from training.trainer import Trainer
from training.loss import Loss



class TrainerBase():
    def __init__(
        self,
        hparams_file:str
    ) -> None:

        # get hyper-parameters and other variables
        self.args = Args(
            file_name=hparams_file
        )

        # initialize taichi
        taichi_init_args = {"arch": ti.cuda,}
        ti.init(**taichi_init_args)

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
            'dataset': self.train_dataset,
        }
        self.model = NGP(**model_config).to(self.args.device)

        # load checkpoint if ckpt path is provided
        if self.args.model.ckpt_path:
            self._loadCheckpoint(ckpt_path=self.args.model.ckpt_path)

    def interfereDensityMap(
            self, 
            res_map:int, 
            height_w:float, 
            num_avg_heights:int,
            tolerance_w:float,
            threshold:float,
    ):
        """
        Evaluate slice density.
        Args:
            res_map: number of samples in each dimension (L); int
            height_w: height of slice in world coordinates (meters); float
            num_avg_heights: number of heights to average over (A); int
            tolerance_w: tolerance in world coordinates (meters); float
            threshold: threshold for density map; float
        Returns:
            density_map: density map of slice; array of shape (L, L)
        """
        # create position grid
        pos_avg = self.createMapPos(
            res_map=res_map,
            height_w=height_w,
            num_avg_heights=num_avg_heights,
            tolerance_w=tolerance_w,
        ) # (L*L*A, 3)

        # interfere density map
        density_map = torch.empty(0).to(self.args.device)
        for density_batch in self._batchifyDensity(
                pos=pos_avg,
                batch_size=self.args.eval.batch_size,
                test_time=True,
            ):
            density_map = torch.cat((density_map, density_batch), dim=0)

        density_map = density_map.detach().cpu().numpy().reshape(-1, num_avg_heights) # (L*L, A)
        density_map = np.nanmax(density_map, axis=1) # (L*L,)
        density_map = density_map.reshape(res_map, res_map) # (L, L)

        # threshold density map
        density_map_thr = np.zeros_like(density_map)
        density_map_thr[density_map < threshold] = 0.0
        density_map_thr[density_map >= threshold] = 1.0

        return density_map, density_map_thr # (L, L), (L, L)
    
    def createMapPos(
            self,
            res_map:int,
            height_w:float,
            num_avg_heights:int,
            tolerance_w:float,
    ):
        """
        Create map positions to evaluate density for different heights.
        Args:
            res_map: number of samples in each dimension (L); int
            height_w: height of slice in world coordinates (meters); float
            num_avg_heights: number of heights to average over (A); int
            tolerance_w: tolerance in world coordinates (meters); float
        Returns:
            pos_avg: map positions for different heights; array of shape (L*L*A, 3)
        """
        # convert distances from meters to cube coordinates
        height_c = self.train_dataset.scene.w2cTransformation(pos=np.array([[0.0, 0.0, height_w]]), copy=True)[0,2]
        tolerance_c = self.train_dataset.scene.w2cTransformation(pos=tolerance_w, only_scale=True, copy=True)

        # create map positions
        pos = torch.linspace(self.test_dataset.scene.w2c_params["cube_min"], self.test_dataset.scene.w2c_params["cube_max"], res_map).to(self.args.device) # (L,)
        m1, m2 = torch.meshgrid(pos, pos) # (L, L), (L, L)
        pos = torch.stack((m1.reshape(-1), m2.reshape(-1)), dim=1) # (L*L, 2)

        # create map positions for different heights
        pos_avg = torch.zeros(res_map*res_map, num_avg_heights, 3).to(self.args.device) # (L*L, A, 3)
        for i, h in enumerate(np.linspace(height_c-tolerance_c, height_c+tolerance_c, num_avg_heights)):
            pos_avg[:,i,:2] = pos
            pos_avg[:,i,2] = h

        return pos_avg.reshape(-1, 3) # (L*L*A, 3)

    def createScanMaps(
            self,
            rays_o_w:np.array,
            depth:np.array,
            scan_angles:np.array,
    ):
        """
        Create scan maps for given rays and depths.
        Args:
            rays_o_w: ray origins in world coordinates (meters); numpy array of shape (N*M, 3)
            depth: depths in wolrd coordinates (meters); numpy array of shape (N*M,)
            scan_angles: scan angles; numpy array of shape (N*M,)
        Returns:
            scan_maps: scan maps; numpy array of shape (N, M, M)
        """
        M = self.args.eval.res_angular
        N = rays_o_w.shape[0] // M
        if rays_o_w.shape[0] % M != 0:
            self.args.logger.error(f"trainer_RH.createScanMaps(): rays_o_w.shape[0]={rays_o_w.shape[0]} % M={M} != 0")
        
        # convert depth to position in world coordinate system and then to map indices
        pos = self.test_dataset.scene.convertDepth2Pos(rays_o=rays_o_w, scan_depth=depth, scan_angles=scan_angles) # (N*M, 2)
        idxs = self.test_dataset.scene.w2idxTransformation(pos=pos, res=M) # (N*M, 2)
        idxs = idxs.reshape(N, M, 2) # (N, M, 2)

        # create scan map
        scan_maps = np.zeros((N, M, M))
        for i in range(N):
            scan_maps[i, idxs[i,:,0], idxs[i,:,1]] = 1.0

        return scan_maps # (N, M, M)
    
    def _loadCheckpoint(
        self, 
        ckpt_path:str
    ):
        """
        Load checkpoint
        Args:
            ckpt_path: path to checkpoint; str
        """
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print("Load checkpoint from %s" % ckpt_path)

    def _saveModel(
        self,
    ):
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
    
    def _batchifyRender(
            self,
            rays_o:torch.Tensor,
            rays_d:torch.Tensor,
            test_time:bool,
            batch_size:int,
    ):
        """
        Batchify rendering process.
        Args:
            rays_o: ray origins; tensor of shape (N, 3)
            rays_d: ray directions; tensor of shape (N, 3)
            test_time: test time rendering; bool
            batch_size: batch size; int 
        Yields:
            results: dict of rendered images of current batch
        """
        # calculate number of batches
        N = rays_o.shape[0]
        if N % batch_size == 0:
            num_batches = N // batch_size
        else:
            num_batches = N // batch_size + 1

        # render rays in batches
        with torch.no_grad() if test_time else nullcontext():
            with alive_bar(num_batches, bar = 'bubbles', receipt=False) as bar:
                for i in range(num_batches):
                    batch_start = i * batch_size
                    batch_end = min((i+1) * batch_size, N)
                    results = render(
                        self.model, 
                        rays_o=rays_o[batch_start:batch_end], 
                        rays_d=rays_d[batch_start:batch_end],
                        test_time=test_time,
                        exp_step_factor=self.args.exp_step_factor,
                    )
                    bar()
                    yield results

    def _batchifyDensity(
            self,
            pos:torch.Tensor,
            test_time:bool,
            batch_size:int,
    ):
        """
        Batchify density rendering process.
        Args:
            pos: ray origins; tensor of shape (N, 3)
            test_time: test time rendering; bool
            batch_size: batch size; int 
        Yields:
            sigmas: density of current batch; tensor of shape (N,)
        """
        # calculate number of batches
        N = pos.shape[0]
        if N % batch_size == 0:
            num_batches = N // batch_size
        else:
            num_batches = N // batch_size + 1

        # render rays in batches
        with torch.no_grad() if test_time else nullcontext():
            with alive_bar(num_batches, bar = 'bubbles', receipt=False) as bar:
                for i in range(num_batches):
                    batch_start = i * batch_size
                    batch_end = min((i+1) * batch_size, N)
                    sigmas = self.model.density(pos[batch_start:batch_end])
                    bar()
                    yield sigmas

    def _step2time(
        self,
        steps:np.array,
    ):
        """
        Convert steps to time by linear interpolating the logs.
        Args:
            steps: steps to convert; array of shape (N,)
        Returns:
            times: times of given steps; array of shape (N,)
        """
        if len(steps) == 0:
            return np.array([])
        
        slope = self.logs['time'][-1] / self.logs['step'][-1]
        return slope * steps
        
        # return linInterpolateArray(
        #     x1=np.array(self.logs['step']),
        #     y1=np.array(self.logs['time']),
        #     x2=steps,
        #     border_condition="nearest"
        # )
    
    def _time2step(
        self,
        times:np.array,
    ):
        """
        Convert time to steps by linear interpolating the logs.
        Args:
            times: times to convert; array of shape (N,)
        Returns:
            steps: steps of given times; array of shape (N,)
        """
        if len(times) == 0:
            return np.array([])
        
        slope = self.logs['step'][-1] / self.logs['time'][-1]
        return slope * times
        
        # return linInterpolateArray(
        #     x1=np.array(self.logs['time']),
        #     y1=np.array(self.logs['step']),
        #     x2=times,
        #     border_condition="nearest"
        # )

