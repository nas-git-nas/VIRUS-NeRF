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
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from contextlib import nullcontext

# from gui import NGPGUI
from datasets import dataset_dict
from datasets.ray_utils import get_rays

from modules.networks import NGP
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model
from helpers.geometric_fcts import findNearestNeighbour
from helpers.data_fcts import linInterpolateArray
from training.metrics_rh import MetricsRH

from modules.occupancy_grid import OccupancyGrid

from training.trainer import Trainer
from training.loss import Loss

class TrainerRH(Trainer):
    def __init__(self, hparams_file) -> None:
        print(f"\n----- START INITIALIZING -----")
        Trainer.__init__(self, hparams_file=hparams_file)

        # loss function
        self.loss = Loss(
            args=self.args,
            rh_scene=self.train_dataset.scene,
            sensors_dict=self.train_dataset.sensors_dict,
        )

        # metrics
        self.metrics = MetricsRH(
            args=self.args,
            rh_scene=self.train_dataset.scene,
            img_wh=self.train_dataset.img_wh,
        )

        # logs
        self.logs = {
            'time': [],
            'step': [],
            'loss': [],
            'color_loss': [],
            'depth_loss': [],
            'rgbd_loss': [],
            'ToF_loss': [],
            'USS_loss': [],
            'USS_close_loss': [],
            'USS_min_loss': [],
            'psnr': [],
            'mnn': [],
        }

        self.occ_grid_class = OccupancyGrid(
            args=self.args,
            grid_size=self.model.grid_size,
            rh_scene=self.train_dataset.scene,
        )

    def train(self):
        """
        Training loop.
        """
        print(f"\n----- START TRAINING -----")
        train_tic = time.time()
        for step in range(self.args.training.max_steps):
            self.model.train()

            data = self.train_dataset(
                batch_size=self.args.training.batch_size,
                sampling_strategy=self.args.training.sampling_strategy,
                origin="nerf",
            )

            direction = data['direction']
            pose = data['pose']

            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                
                if step % self.args.occ_grid.update_interval == 0:

                    if self.args.occ_grid.grid_type == 'nerf':
                        self.model.updateNeRFGrid(
                            density_threshold=0.01 * MAX_SAMPLES / 3**0.5,
                            warmup=step < self.args.occ_grid.warmup_steps,
                        )
                    elif self.args.occ_grid.grid_type == 'occ':
                        self.model.updateOccGrid(
                            density_threshold= 0.5,
                        )
                    else:
                        print(f"ERROR: NGP:__init__: grid_type {self.args.occ_grid.grid_type} not implemented")
                    

                # get rays and render image
                rays_o, rays_d = get_rays(direction, pose)
                results = render(
                    self.model, 
                    rays_o, 
                    rays_d,
                    exp_step_factor=self.args.exp_step_factor,
                )

                # calculate loss
                loss, loss_dict = self.loss(
                    results=results,
                    data=data,
                    return_loss_dict=True, # TODO: optimize
                )
                # loss, color_loss, depth_loss = self.lossFunc(results=results, data=data, step=step)
                if self.args.training.distortion_loss_w > 0:
                    loss += self.args.training.distortion_loss_w * distortion_loss(results).mean()

            # backpropagate and update weights
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.scheduler.step()

            # evaluation
            eval_tic = time.time()
            self._evaluateStep(
                results=results, 
                data=data, 
                step=step, 
                loss_dict=loss_dict,
                tic=train_tic
            )
            self._plotOccGrid(
                step=step,
            )
            eval_toc = time.time()
            train_tic += eval_toc - eval_tic # subtract evaluation time from training time

        self.saveModel()

    def evaluate(self):
        """
        Evaluate NeRF on test set.
        """
        print(f"\n----- START EVALUATING -----")
        self.model.eval()

        # get indices of all test points and of one particular sensor
        img_idxs = np.arange(len(self.test_dataset))
        img_idxs_sensor = self.test_dataset.getIdxFromSensorName(df=self.test_dataset.df, sensor_name="RGBD_1")

        # keep only a certain number of points
        if self.args.eval.num_color_pts != "all":
            idxs_temp = np.random.randint(0, len(img_idxs), self.args.eval.num_color_pts)
            # idxs_temp = np.linspace(0, len(img_idxs)-1, self.args.eval.num_color_pts, dtype=int)
            img_idxs = img_idxs[idxs_temp]

        if self.args.eval.num_depth_pts != "all":
            idxs_temp = np.linspace(0, len(img_idxs_sensor)-1, self.args.eval.num_depth_pts, dtype=int)
            img_idxs_sensor = img_idxs_sensor[idxs_temp]

        # evaluate color and depth
        metrics_dict = self.evaluateColor(img_idxs=img_idxs)
        depth_metrics, data_w = self.evaluateDepth(img_idxs=img_idxs_sensor)
        metrics_dict.update(depth_metrics)

        # print and save metrics
        print(
            f"evaluation: " \
            + f"psnr_avg={np.round(metrics_dict['psnr'],2)} | " \
            + f"ssim_avg={metrics_dict['ssim']:.3} | " \
            + f"depth_mae={metrics_dict['mae']:.3} | " \
            + f"depth_mare={metrics_dict['mare']:.4} | " \
            + f"depth_mnn={metrics_dict['mnn']:.3} | " \
        )
        metrics_df = metrics_dict.copy()
        del metrics_df['nn_dists']
        metrics_df = pd.DataFrame(metrics_df, index=[0])
        metrics_df.to_csv(os.path.join(self.args.save_dir, "metrics.csv"), index=False)

        # create plots
        self.evaluatePlot(
            data_w=data_w, 
            metrics_dict=metrics_dict
        )
        self._plotLosses()

    @torch.no_grad()
    def evaluateColor(
            self,
            img_idxs:np.array,
    ):
        """
        Evaluate color error.
        Args:
            img_idxs: image indices; array of int (N,)
        Returns:
            metrics_dict: dict of metrics
        """
        W, H = self.test_dataset.img_wh
        N = img_idxs.shape[0]

        # repeat image indices and pixel indices
        img_idxs = img_idxs.repeat(W*H) # (N*W*H,)
        pix_idxs = np.tile(np.arange(W*H), N) # (N*W*H,)

        # get poses, direction and color ground truth
        poses = self.test_dataset.poses[img_idxs]
        directions = self.test_dataset.directions[pix_idxs]
        rgb_gt = self.test_dataset.rays[img_idxs, pix_idxs][:, :3]

        # calculate rays
        rays_o, rays_d = get_rays(
            directions=directions, 
            c2w=poses
        ) # (N*W*H, 3), (N*W*H, 3)

        # render rays to get color
        rgb = torch.empty(0, 3).to(self.args.device)
        depth = torch.empty(0).to(self.args.device)
        for results in self._batchifyRender(
                rays_o=rays_o,
                rays_d=rays_d,
                test_time=True,
                batch_size=self.args.eval.batch_size,
            ):
            rgb = torch.cat((rgb, results['rgb']), dim=0)
            depth = torch.cat((depth, results['depth']), dim=0)

        # calculate metrics
        metrics_dict = self.metrics.evaluate(
            data={ 'rgb': rgb, 'rgb_gt': rgb_gt },
            eval_metrics=['psnr', 'ssim'],
            convert_to_world_coords=False,
            copy=True,
        )

        # save example image
        test_idx = 0 # TODO: customize
        print(f"Saving test image {test_idx} to disk")
        
        rgb_path = os.path.join(self.args.save_dir, f'rgb_{test_idx:03d}.png')
        rgb_img = rearrange(rgb[:H*W].cpu().numpy(),'(h w) c -> h w c', h=H) # TODO: optimize
        rgb_img = (rgb_img * 255).astype(np.uint8)
        imageio.imsave(rgb_path, rgb_img)

        depth_path = os.path.join(self.args.save_dir, f'depth_{test_idx:03d}.png')
        depth_img = rearrange(depth[:H*W].cpu().numpy(), '(h w) -> h w', h=H) # TODO: optimize
        depth_img = depth2img(depth_img)
        imageio.imsave(depth_path, depth_img)

        return metrics_dict

    @torch.no_grad()
    def evaluateDepth(
            self, 
            img_idxs:np.array,
    ) -> dict:
        """
        Evaluate depth error.
        Args:
            img_idxs: image indices; array of int (N,)
        Returns:
            metrics_dict: dict of metrics
            data_w: dict of data in world coordinates
        """
        # create scan rays for averaging over different heights
        rays_o, rays_d = self.createScanRays(
            img_idxs=img_idxs,
            res_angular=self.args.eval.res_angular,
            num_avg_heights=self.args.eval.num_avg_heights,
            height_tolerance=self.args.eval.height_tolerance,
        ) # (N*M*A, 3), (N*M*A, 3)

        # render rays to get depth
        depths = torch.empty(0).to(self.args.device)
        for results in self._batchifyRender(
                rays_o=rays_o,
                rays_d=rays_d,
                test_time=True,
                batch_size=self.args.eval.batch_size,
            ):
            depths = torch.cat((depths, results['depth']), dim=0)

        # average dpeth over different heights
        depths = depths.detach().cpu().numpy().reshape(-1, self.args.eval.num_avg_heights) # (N*M, A)
        depth = np.nanmean(depths, axis=1) # (N*M,)

        # create scan rays (without averaging over different heights)
        rays_o, rays_d = self.createScanRays(
            img_idxs=img_idxs,
            res_angular=self.args.eval.res_angular,
            num_avg_heights=1,
            height_tolerance=0.0,
        ) # (N*M, 3), (N*M, 3)
        rays_o = rays_o.detach().cpu().numpy() # (N*M, 3)
        rays_d = rays_d.detach().cpu().numpy() # (N*M, 3)

        # get ground truth depth
        scan_map_gt, depth_gt, scan_angles = self.test_dataset.scene.getSliceScan(
            res=self.args.eval.res_map, 
            rays_o=rays_o, 
            rays_d=rays_d, 
            rays_o_in_world_coord=False, 
            height_tolerance=self.args.eval.height_tolerance
        )

        # convert depth to world coordinates (meters)
        depth_w = self.test_dataset.scene.c2wTransformation(pos=depth, only_scale=True, copy=True)
        depth_w_gt = self.test_dataset.scene.c2wTransformation(pos=depth_gt, only_scale=True, copy=True)
        rays_o_w = self.test_dataset.scene.c2wTransformation(pos=rays_o, copy=True) # (N*M, 3)
        data_w = {
            'depth': depth_w,
            'depth_gt': depth_w_gt,
            'rays_o': rays_o_w,
            'scan_angles': scan_angles,
            'scan_map_gt': scan_map_gt,
        }

        # calculate mean squared depth error
        metrics_dict = self.metrics.evaluate(
            data=data_w,
            eval_metrics=['rmse', 'mae', 'mare', 'nn'],
            convert_to_world_coords=False,
            copy=True,
            num_test_pts=len(img_idxs),
        )

        return metrics_dict, data_w
    
    @torch.no_grad()
    def evaluatePlot(
            self,
            data_w:dict,
            metrics_dict:dict,
    ):
        """
        Plot scan and density maps.
        Args:
            data_w: data dictionary in world coordinates
            metrics_dict: metrics dictionary
        """
        M = self.args.eval.res_angular
        N = data_w['depth'].shape[0] // M
        if data_w['depth'].shape[0] % M != 0:
            print(f"ERROR: trainer_RH.evaluatePlot(): data_w['depth'].shape[0]={data_w['depth'].shape[0]} "
                  f"should be a multiple of M={M}")
        
        # downsample data
        depth_w = data_w['depth'].reshape((N, M)) # (N, M)
        rays_o_w = data_w['rays_o'].reshape((N, M, 3)) # (N, M, 3)
        scan_angles = data_w['scan_angles'].reshape((N, M)) # (N, M)
        nn_dists = metrics_dict['nn_dists'] # (N, M)
        if self.args.eval.num_plot_pts > N:
            print(f"WARNING: trainer_RH.evaluatePlot(): num_plot_pts={self.args.eval.num_plot_pts} "
                  f"should be smaller or equal than N={N}")
            self.args.eval.num_plot_pts = N
        elif self.args.eval.num_plot_pts < N:
            idxs_temp = np.linspace(0, depth_w.shape[0]-1, self.args.eval.num_plot_pts, dtype=int)
            depth_w = depth_w[idxs_temp]
            rays_o_w = rays_o_w[idxs_temp]
            scan_angles = scan_angles[idxs_temp]
            nn_dists = nn_dists[idxs_temp]  
        depth_w = depth_w.flatten() # (N*M,)
        rays_o_w = rays_o_w.reshape((-1, 3)) # (N*M, 3)
        scan_angles = scan_angles.flatten() # (N*M,)

        # create scan maps
        scan_maps = self._createScanMaps(
            rays_o_w=rays_o_w,
            depth=depth_w,
            scan_angles=scan_angles,
        ) # (N, L, L)
        scan_map_gt = data_w['scan_map_gt'] # (L, L)

        # create density maps
        density_map_gt = self.test_dataset.scene.getSliceMap(
            height=np.mean(rays_o_w[:,2]), 
            res=self.args.eval.res_map, 
            height_tolerance=self.args.eval.height_tolerance, 
            height_in_world_coord=True
        ) # (L, L)
        _, density_map = self.interfereDensityMap(
            res_map=self.args.eval.res_map,
            height_w=np.mean(rays_o_w[:,2]),
            num_avg_heights=self.args.eval.num_avg_heights,
            tolerance_w=self.args.eval.height_tolerance,
            threshold=self.args.eval.density_map_thr,
        ) # (L, L)

        # create combined maps
        scan_maps_comb = np.zeros((self.args.eval.num_plot_pts,self.args.eval.res_map,self.args.eval.res_map))
        for i in range(self.args.eval.num_plot_pts):
            scan_maps_comb[i] = scan_map_gt + 2*scan_maps[i]
        density_map_comb = density_map_gt + 2*density_map

        # plot
        fig, axes = plt.subplots(ncols=1+self.args.eval.num_plot_pts, nrows=4, figsize=(9,9))

        scale = self.args.model.scale
        extent = self.test_dataset.scene.c2wTransformation(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
        extent = extent.T.flatten()

        ax = axes[0,0]
        ax.imshow(density_map_gt.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(density_map_comb))
        ax.set_ylabel(f'GT', weight='bold')
        ax.set_title(f'Density', weight='bold')
        ax.set_xlabel(f'x [m]')

        ax = axes[1,0]
        ax.imshow(2*density_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(density_map_comb))
        ax.set_ylabel(f'NeRF', weight='bold')
        ax.set_xlabel(f'x [m]')

        ax = axes[2,0]
        ax.imshow(density_map_comb.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(density_map_comb))
        ax.set_ylabel(f'GT + NeRF', weight='bold')
        ax.set_xlabel(f'x [m]')

        fig.delaxes(axes[3,0])
        
        rays_o_w = rays_o_w.reshape((-1, self.args.eval.res_angular, 3))
        for i in range(self.args.eval.num_plot_pts):
            ax = axes[0,i+1]
            ax.imshow(scan_map_gt.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(scan_maps_comb[i]))
            ax.scatter(rays_o_w[i,0,0], rays_o_w[i,0,1], c='w', s=5)
            ax.scatter(rays_o_w[:,0,0], rays_o_w[:,0,1], c='w', s=5, alpha=0.1)
            ax.set_title(f'Scan {i+1}', weight='bold')
            ax.set_xlabel(f'x [m]')
            ax.set_ylabel(f'y [m]')

            ax = axes[1,i+1]
            ax.imshow(2*scan_maps[i].T,origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(scan_maps_comb[i]))
            # for i in range(0, rays_o_w.shape[0], nb_pts_step):
            #     for j in range(depth_pos_w.shape[1]):
            #         ax.plot([rays_o_w[i,j,0], depth_pos_w[i,j,0]], [rays_o_w[i,j,1], depth_pos_w[i,j,1]], c='w', linewidth=0.1)
            ax.scatter(rays_o_w[i,0,0], rays_o_w[i,0,1], c='w', s=5)
            ax.scatter(rays_o_w[:,0,0], rays_o_w[:,0,1], c='w', s=5, alpha=0.1)
            ax.set_xlabel(f'x [m]')
            ax.set_ylabel(f'y [m]')
            
            ax = axes[2,i+1]
            ax.imshow(scan_maps_comb[i].T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(scan_maps_comb[i]))
            # for i in range(0, rays_o_w.shape[0], nb_pts_step):
            #     for j in range(depth_pos_w.shape[1]):
            #         ax.plot([rays_o_w[i,j,0], depth_pos_w[i,j,0]], [rays_o_w[i,j,1], depth_pos_w[i,j,1]], c='w', linewidth=0.1)
            ax.scatter(rays_o_w[i,0,0], rays_o_w[i,0,1], c='w', s=5)
            ax.scatter(rays_o_w[:,0,0], rays_o_w[:,0,1], c='w', s=5, alpha=0.1)
            ax.set_xlabel(f'x [m]')
            ax.set_ylabel(f'y [m]')

            ax = axes[3,i+1]
            ax.hist(nn_dists[i], bins=50)
            ax.vlines(np.nanmean(nn_dists[i]), ymin=0, ymax=20, colors='r', linestyles='dashed', label=f'avg.={np.nanmean(nn_dists[i]):.2f}')
            if i == 0:
                ax.set_ylabel(f'Nearest Neighbour', weight='bold')
            else:
                ax.set_ylabel(f'# elements')
            ax.set_xlim([0, np.nanmax(nn_dists)])
            ax.set_ylim([0, 25])
            ax.set_xlabel(f'distance [m]')
            ax.legend()
            ax.set_box_aspect(1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "maps.pdf"))
        plt.savefig(os.path.join(self.args.save_dir, "maps.png"))

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

        return density_map, density_map_thr # (L, L)
    
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

    def createScanRays(
            self,
            img_idxs:np.array,
            res_angular:int,
            num_avg_heights:int=1,
            height_tolerance:float=0.1,
    ):
        """
        Create scan rays for gievn image indices.
        Args:
            img_idxs: image indices; array of int (N,)
            res_angular: number of angular samples (M); int
            num_avg_heights: number of heights to average over (A); int
            height_tolerance: height tolerance in world coordinates (meters); float
        Returns:
            rays_o: ray origins; array of shape (N*M*A, 3)
            rays_d: ray directions; array of shape (N*M*A, 3)
        """
        # get rays
        rays_o = self.test_dataset.poses[img_idxs, :3, 3].detach().clone() # (N, 3)
        rays_o = torch.repeat_interleave(rays_o, res_angular, dim=0) # (N*M, 3)

        # get directions
        rays_d = torch.linspace(-np.pi, np.pi-2*np.pi/res_angular, res_angular, 
                                dtype=torch.float32, device=self.args.device) # (M,)
        rays_d = torch.stack((torch.cos(rays_d), torch.sin(rays_d), torch.zeros_like(rays_d)), axis=1) # (M, 3)
        rays_d = rays_d.repeat(len(img_idxs), 1) # (N*M, 3)

        if num_avg_heights == 1:
            return rays_o, rays_d
        
        # convert height tolerance to cube coordinates
        h_tol_c = self.test_dataset.scene.w2cTransformation(pos=height_tolerance, only_scale=True, copy=True)

        # get rays for different heights
        rays_o_avg = torch.zeros(len(img_idxs)*res_angular, num_avg_heights, 3).to(self.args.device) # (N*M, A, 3)
        rays_d_avg = torch.zeros(len(img_idxs)*res_angular, num_avg_heights, 3).to(self.args.device) # (N*M, A, 3)   
        for i, h in enumerate(np.linspace(-h_tol_c, h_tol_c, num_avg_heights)):
            h_tensor = torch.tensor([0.0, 0.0, h], dtype=torch.float32, device=self.args.device)
            rays_o_avg[:,i,:] = rays_o + h_tensor
            rays_d_avg[:,i,:] = rays_d

        return rays_o_avg.reshape(-1, 3), rays_d_avg.reshape(-1, 3) # (N*M*A, 3), (N*M*A, 3)
    
    def _createScanMaps(
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
            print(f"ERROR: trainer_RH._createScanMaps(): rays_o.shape[0]={rays_o_w.shape[0]} % M={M} != 0")
        
        # convert depth to position in world coordinate system and then to map indices
        pos = self.test_dataset.scene.convertDepth2Pos(rays_o=rays_o_w, scan_depth=depth, scan_angles=scan_angles) # (N*M, 2)
        idxs = self.test_dataset.scene.w2idxTransformation(pos=pos, res=M) # (N*M, 2)
        idxs = idxs.reshape(N, M, 2) # (N, M, 2)

        # create scan map
        scan_maps = np.zeros((N, M, M))
        for i in range(N):
            scan_maps[i, idxs[i,:,0], idxs[i,:,1]] = 1.0

        return scan_maps # (N, M, M)
    
    @torch.no_grad()
    def _evaluateStep(
            self, 
            results:dict, 
            data:dict, 
            step:int, 
            loss_dict:dict,
            tic:time.time,
    ):
        """
        Print statistics about the current training step.
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
            step: current training step; int
            loss_dict: dict of sub-losses
            tic: training starting time; time.time()
        """
        # log parameters
        self.logs['time'].append(time.time()-tic)
        self.logs['step'].append(step+1)
        self.logs['loss'].append(loss_dict['total'])
        self.logs['color_loss'].append(loss_dict['color'])
        self.logs['depth_loss'].append(loss_dict['depth'])
        if "rgbd" in loss_dict:
            self.logs['rgbd_loss'].append(loss_dict['rgbd'])
        if "ToF" in loss_dict:
            self.logs['ToF_loss'].append(loss_dict['ToF'])
        if "USS" in loss_dict:
            self.logs['USS_loss'].append(loss_dict['USS'])
            self.logs['USS_close_loss'].append(loss_dict['USS_close'])
            self.logs['USS_min_loss'].append(loss_dict['USS_min'])
        self.logs['psnr'].append(np.nan)
        self.logs['mnn'].append(np.nan)

        # make intermediate evaluation
        if step % self.args.eval.eval_every_n_steps == 0:
            # evaluate color and depth of one random image
            img_idxs = np.array(np.random.randint(0, len(self.test_dataset), size=8))
            depth_metrics, data_w = self.evaluateDepth(img_idxs=img_idxs)

            # calculate peak-signal-to-noise ratio
            mse = F.mse_loss(results['rgb'], data['rgb'])
            psnr = -10.0 * torch.log(mse) / np.log(10.0)

            self.logs['psnr'][-1] = psnr.item()
            self.logs['mnn'][-1] = depth_metrics['mnn']
            print(
                f"time={(time.time()-tic):.2f}s | "
                f"step={step} | "
                f"lr={(self.optimizer.param_groups[0]['lr']):.5f} | "
                f"loss={loss_dict['total']:.4f} | "
                f"color_loss={loss_dict['color']:.4f} | "
                f"depth_loss={loss_dict['depth']:.4f} | "
                f"psnr={psnr:.2f} | "
                f"depth_mnn={(depth_metrics['mnn']):.3f} | "
            )

        # # calculate peak-signal-to-noise ratio
        # with torch.no_grad():
        #     mse = F.mse_loss(results['rgb'], data['rgb'])
        #     psnr = -10.0 * torch.log(mse) / np.log(10.0)
        #     error, _, _, _, _, _ = self.evaluateDepth()

        # # print statistics
        # print(
        #     f"time={(time.time()-tic):.2f}s | "
        #     f"step={step} | "
        #     f"psnr={psnr:.2f} | "
        #     f"loss={loss:.4f} | "
        #     f"color_loss={color_loss:.4f} | "
        #     f"depth_loss={depth_loss:.4f} | "
        #     # number of rays
        #     f"rays={len(data['rgb'])} | "
        #     # ray marching samples per ray (occupied space on the ray)
        #     f"rm_s={results['rm_samples'] / len(data['rgb']):.1f} | "
        #     # volume rendering samples per ray 
        #     # (stops marching when transmittance drops below 1e-4)
        #     f"vr_s={results['vr_samples'] / len(data['rgb']):.1f} | "
        #     f"lr={(self.optimizer.param_groups[0]['lr']):.5f} | "
        #     f"depth_mae={error['depth_mae']:.3f} | "
        #     f"depth_mare={error['depth_mare']:.3f} | "
        # )

    def _plotLosses(
        self,
    ):
        """
        Plot losses, mean-nearest-neighbour distance and peak signal-to-noise-ratio.
        """
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,8))

        # plot losses
        ax = axes[0]
        ax.plot(self.logs['step'], self.logs['loss'], label='total')
        ax.plot(self.logs['step'], self.logs['color_loss'], label='color')
        ax.plot(self.logs['step'], self.logs['depth_loss'], label='depth')
        if "rgbd" in self.logs:
            ax.plot(self.logs['step'], self.logs['rgbd_loss'], label='rgbd')
        if "ToF" in self.logs:
            ax.plot(self.logs['step'], self.logs['ToF_loss'], label='ToF')
        if "USS" in self.logs:
            ax.plot(self.logs['step'], self.logs['USS_loss'], label='USS')
            ax.plot(self.logs['step'], self.logs['USS_close_loss'], label='USS_close')
            ax.plot(self.logs['step'], self.logs['USS_min_loss'], label='USS_min')
        ax.set_ylabel('loss')
        ax.set_ylim([0, 1.0])

        ax.set_xlabel('step')
        secax = ax.secondary_xaxis(
            location='top', 
            functions=(self._step2time, self._time2step),
        )
        secax.set_xlabel('time [s]')
        ax.legend()
        ax.set_title('Losses')

        # plot mnn and psnr 
        if 'mnn' in self.logs and 'psnr' in self.logs:
            ax = axes[1]
            not_nan = ~np.isnan(self.logs['mnn'])
            lns1 = ax.plot(np.array(self.logs['step'])[not_nan], np.array(self.logs['mnn'])[not_nan], c="blue", label='mnn')
            ax.set_ylabel('mnn')
            ax.set_ylim([0, 0.5])
            ax.yaxis.label.set_color('blue') 
            ax.tick_params(axis='y', colors='blue')

            ax2 = ax.twinx()
            not_nan = ~np.isnan(self.logs['psnr'])
            lns2 = ax2.plot(np.array(self.logs['step'])[not_nan], np.array(self.logs['psnr'])[not_nan], label='psnr', color='green')
            ax2.set_ylabel('psnr')
            ax2.yaxis.label.set_color('green') 
            ax2.tick_params(axis='y', colors='green')

            ax.set_xlabel('step')
            secax = ax.secondary_xaxis(
                location='top', 
                functions=(self._step2time, self._time2step),
            )
            secax.set_xlabel('time [s]')
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
            ax.set_title('Metrics')

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "losses.pdf"))
        plt.savefig(os.path.join(self.args.save_dir, "losses.png"))

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
    
    def _plotOccGrid(
            self,
            step,
    ):
        if step % self.args.occ_grid.update_interval != 0:
            return

        height_w = 1.0
        height_c = self.train_dataset.scene.w2cTransformation(pos=np.array([[0.0, 0.0, height_w]]), copy=False)[0,2]
        if self.args.occ_grid.grid_type == "occ": # TODO: remove after debugging
            height_c = self.model.occupancy_grid.height_c.detach().cpu().numpy()
            height_w = self.train_dataset.scene.c2wTransformation(pos=np.array([[0.0, 0.0, height_c]]), copy=False)[0,2]

        # convert height from cube to occupancy grid coordinates
        height_o = self.model.occupancy_grid.c2oCoordinates(
            pos_c=height_c,
        )

        occ_3d_grid = self.model.occupancy_grid.getOccupancyCartesianGrid()
        bin_3d_grid = self.model.occupancy_grid.getBinaryCartesianGrid(
            threshold=0.5,
        )
        bitfield = self.model.occupancy_grid.getBitfield()
        bin_morton_grid = self.model.occupancy_grid.bitfield2morton(
            bin_bitfield=bitfield,
        )
        bin_3d_recovery = self.model.occupancy_grid.morton2cartesian(
            grid_morton=bin_morton_grid,
        )

        # TODO: optimize remove
        if not torch.allclose(bin_3d_grid, bin_3d_recovery):
            print(f"ERROR: NGP:updateOccGrid: bin_3d_grid and bin_3d_recovery are not the same")

        # convert from 3D to 2D
        occ_2d_grid = occ_3d_grid[:,:,height_o]
        bin_2d_grid = bin_3d_grid[:,:,height_o]
        bin_2d_recovery = bin_3d_recovery[:,:,height_o]

        # convert from tensor to array
        occ_2d_grid = occ_2d_grid.detach().clone().cpu().numpy()
        bin_2d_grid = bin_2d_grid.detach().clone().cpu().numpy()
        bin_2d_recovery = bin_2d_recovery.detach().clone().cpu().numpy()

        # create density maps
        density_map_gt = self.test_dataset.scene.getSliceMap(
            height=height_w, 
            res=occ_2d_grid.shape[0], 
            height_tolerance=self.args.eval.height_tolerance, 
            height_in_world_coord=True
        ) # (L, L)
        density_map, density_map_thr = self.interfereDensityMap(
            res_map=occ_2d_grid.shape[0],
            height_w=height_w,
            num_avg_heights=1,
            tolerance_w=0.0,
            threshold=0.01 * MAX_SAMPLES / 3**0.5,
        ) # (L, L)

        # plot occupancy grid
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(9,6))
        scale = self.args.model.scale
        extent = self.test_dataset.scene.c2wTransformation(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
        extent = extent.T.flatten()

        ax = axes[0,0]
        im = ax.imshow(density_map_gt.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=1)
        ax.set_ylabel(f'y [m]')
        ax.set_title(f'Ground Truth')
        fig.colorbar(im, ax=ax)

        ax = axes[0,1]
        im = ax.imshow(density_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=10 * (0.01 * MAX_SAMPLES / 3**0.5))
        ax.set_ylabel(f'y [m]')
        ax.set_title(f'NeRF density')
        fig.colorbar(im, ax=ax)

        ax = axes[0,2]
        im = ax.imshow(density_map_thr.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'NeRF binary')
        fig.colorbar(im, ax=ax)

        fig.delaxes(axes[1,0])

        ax = axes[1,1]
        im = ax.imshow(occ_2d_grid.T, origin='lower', cmap='viridis', extent=extent, vmin=0, vmax=1)
        ax.set_xlabel(f'x [m]')
        ax.set_ylabel(f'y [m]')
        ax.set_title(f'OccGrid density')
        fig.colorbar(im, ax=ax)

        ax = axes[1,2]
        im = ax.imshow(bin_2d_grid.T, origin='lower', cmap='viridis', extent=extent, vmin=0, vmax=1)
        ax.set_ylabel(f'y [m]')
        ax.set_title(f'OccGrid binary')
        fig.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "occ_grid"+str(step)+".png"))









    # def lossFunc(self, results, data, step):
    #     """
    #     Loss function for training
    #     Args:
    #         results: dict of rendered images
    #             'opacity': sum(transmittance*alpha); array of shape: (N,)
    #             'depth': sum(transmittance*alpha*t_i); array of shape: (N,)
    #             'rgb': sum(transmittance*alpha*rgb_i); array of shape: (N, 3)
    #             'total_samples': total samples for all rays; int
    #             where   transmittance = exp( -sum(sigma_i * delta_i) )
    #                     alpha = 1 - exp(-sigma_i * delta_i)
    #                     delta_i = t_i+1 - t_i
    #         data: dict of ground truth images
    #             'img_idxs': image indices; array of shape (N,) or (1,) if same image
    #             'pix_idxs': pixel indices; array of shape (N,)
    #             'pose': poses; array of shape (N, 3, 4)
    #             'direction': directions; array of shape (N, 3)
    #             'rgb': pixel colours; array of shape (N, 3)
    #             'depth': pixel depths; array of shape (N,)
    #         step: current training step; int
    #     Returns:
    #         total_loss: loss value; float
    #         colour_loss: colour loss value; float
    #         depth_loss: depth loss value; float
    #     """
    #     colour_loss = self._colorLoss(results=results, data=data)
    #     depth_loss = self._depthLoss(results=results, data=data, step=step)
        
    #     depth_loss = depth_loss * self.args.training.depth_loss_w
    #     total_loss = colour_loss + depth_loss
    #     return total_loss, colour_loss, depth_loss


    # def _colorLoss(self, results, data):
    #     """
    #     Loss function for training
    #     Args:
    #         results: dict of rendered images
    #         data: dict of ground truth images
    #     Returns:
    #         colour_loss: colour loss value; float
    #     """
    #     return F.mse_loss(results['rgb'], data['rgb'])
    
    # def _depthLoss(self, results, data, step):
    #     """
    #     Loss function for training
    #     Args:
    #         results: dict of rendered images
    #         data: dict of ground truth images
    #         step: current training step; int
    #     Returns:
    #         depth_loss: depth loss value; float
    #     """
    #     # val_idxs = ~torch.isnan(data['depth'])
    #     # depth_loss = self.args.training.depth_loss_w * F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])
    #     # if torch.all(torch.isnan(depth_loss)):
    #     #     print("WARNING: trainer:lossFunc: depth_loss is nan, set to 0.")
    #     #     depth_loss = 0

    #     if self.args.rh.sensor_model == 'RGBD' or self.args.rh.sensor_model == 'ToF':
    #         val_idxs = ~torch.isnan(data['depth'])
    #         return F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])

    #     if self.args.rh.sensor_model == 'USS':
    #         # get minimum depth per image for batch 
    #         imgs_depth_min, weights = self.train_dataset.sensor_model.updateDepthMin(
    #             results=results,
    #             data=data,
    #             step=step,
    #         ) # (num_test_imgs,), (num_test_imgs,)
    #         depths_min = imgs_depth_min[data['img_idxs']] # (N,)
    #         weights = weights[data['img_idxs']] # (N,)

    #         # mask data
    #         depth_tolerance = self.train_dataset.scene.w2cTransformation(pos=0.03, only_scale=True, copy=True)
    #         depth_tolerance = torch.tensor(depth_tolerance, device=self.args.device, dtype=torch.float32)
    #         uss_mask = ~torch.isnan(data['depth'])
    #         depth_mask = results['depth'] < depths_min + depth_tolerance  
    #         close_mask = results['depth'] < data['depth'] - depth_tolerance  

    #         # calculate loss
    #         depth_loss = torch.tensor(0.0, device=self.args.device, dtype=torch.float32)

    #         depth_data = data['depth'][uss_mask & depth_mask]
    #         w = weights[uss_mask & depth_mask]
    #         depth_results = results['depth'][uss_mask & depth_mask]
    #         if torch.any(uss_mask & depth_mask):
    #             min_loss = torch.mean(w * (depth_results-depth_data)**2)
    #             depth_loss += min_loss
    #             if step%25 == 0:
    #                 print(f"min_loss: {min_loss}")

    #         depth_data = data['depth'][uss_mask & close_mask]
    #         depth_results = results['depth'][uss_mask & close_mask]
    #         if torch.any(uss_mask & close_mask):
    #             close_loss = torch.mean((depth_results-depth_data)**2)
    #             depth_loss += close_loss
    #             if step%25 == 0:
    #                 print(f"close_loss: {close_loss}")

    #         if step%25 == 0:
    #             print(f"depth mask sum: {torch.sum(uss_mask & depth_mask)}, close mask sum: {torch.sum(uss_mask & close_mask)}, weights mean: {torch.mean(weights)}")
            
    #         return depth_loss
        
    #     return torch.tensor(0.0, device=self.args.device, dtype=torch.float32)



# def evaluateColor(
#             self,
#             img_idxs:np.array,
#     ):
#         progress_bar = tqdm.tqdm(total=len(self.test_dataset), desc=f'evaluating: ')

#         w, h = self.test_dataset.img_wh
#         directions = self.test_dataset.directions
#         test_psnrs = []
#         test_ssims = []
#         for i, idx in enumerate(img_idxs):
#             progress_bar.update()
#             test_data = self.test_dataset[idx]

#             rgb_gt = test_data['rgb']
#             poses = test_data['pose']

#             with torch.autocast(device_type='cuda', dtype=torch.float16):
#                 # get rays
#                 rays_o, rays_d = get_rays(directions, poses)
#                 # render image
#                 results = render(
#                     self.model, 
#                     rays_o, 
#                     rays_d,
#                     test_time=True,
#                     exp_step_factor=self.args.exp_step_factor,
#                 )


#             # TODO: get rid of this
#             rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
#             rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
#             # get psnr
#             self.val_psnr(rgb_pred, rgb_gt)
#             test_psnrs.append(self.val_psnr.compute())
#             self.val_psnr.reset()
#             # get ssim
#             self.val_ssim(rgb_pred, rgb_gt)
#             test_ssims.append(self.val_ssim.compute())
#             self.val_ssim.reset()

#             # save test image to disk
#             if i == 0:
#                 print(f"Saving test image {idx} to disk")
#                 test_idx = test_data['img_idxs']
#                 # TODO: get rid of this
#                 rgb_pred = rearrange(
#                     results['rgb'].cpu().numpy(),
#                     '(h w) c -> h w c',
#                     h=h
#                 )
#                 rgb_pred = (rgb_pred * 255).astype(np.uint8)
#                 depth = depth2img(
#                     rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
#                 imageio.imsave(
#                     os.path.join(
#                         self.args.save_dir, 
#                         f'rgb_{test_idx:03d}_'+str(idx)+'.png'
#                         ),
#                     rgb_pred
#                 )
#                 imageio.imsave(
#                     os.path.join(
#                         self.args.save_dir, 
#                         f'depth_{test_idx:03d}.png'
#                     ),
#                     depth
#                 )

#         progress_bar.close()
#         test_psnr_avg = sum(test_psnrs) / len(test_psnrs)
#         test_ssim_avg = sum(test_ssims) / len(test_ssims)


    # def evaluateSlice(self, res, height_w, tolerance_w):
    #     """
    #     Evaluate slice density.
    #     Args:
    #         res: number of samples in each dimension; int
    #         height_w: height of slice in world coordinates (meters); float
    #         tolerance_w: tolerance in world coordinates (meters); float
    #     Returns:
    #         density: density map of slice; array of shape (res,res)
    #     """
    #     # convert distances from meters to cube coordinates
    #     height_c = self.train_dataset.scene.w2cTransformation(pos=np.array([[0.0, 0.0, height_w]]), copy=True)[0,2]
    #     tolerance_c = self.train_dataset.scene.w2cTransformation(pos=tolerance_w, only_scale=True, copy=True)

    #     slice_pts = torch.linspace(self.test_dataset.scene.w2c_params["cube_min"], self.test_dataset.scene.w2c_params["cube_max"], res) # (slice_res,)
    #     m1, m2 = torch.meshgrid(slice_pts, slice_pts) # (slice_res,slice_res), (slice_res,slice_res)
    #     slice_pts = torch.stack((m1.reshape(-1), m2.reshape(-1)), dim=1) # (slice_res*slice_res, 2)


    #     # estimate density of slice
    #     density = []
    #     for h in np.linspace(height_c-tolerance_c, height_c+tolerance_c, 10):         
    #         x = torch.cat((slice_pts, h*torch.ones(res*res,1)), dim=1) # (slice_res*slice_res, 3)
    #         sigmas = self.model.density(x) # (slice_res*slice_res,3)
    #         sigmas = sigmas.reshape(res, res).cpu().detach().numpy() # (slice_res,slice_res)
    #         density.append(sigmas)
    #     density = np.array(density).mean(axis=0)

    #     return density

    # def _printStats(self, results, data, step, loss, color_loss, depth_loss, tic):
    #     """
    #     Print statistics about the current training step.
    #     Args:
    #         results: dict of rendered images
    #             'opacity': sum(transmittance*alpha); array of shape: (N,)
    #             'depth': sum(transmittance*alpha*t__i); array of shape: (N,)
    #             'rgb': sum(transmittance*alpha*rgb_i); array of shape: (N, 3)
    #             'total_samples': total samples for all rays; int
    #             where   transmittance = exp( -sum(sigma_i * delta_i) )
    #                     alpha = 1 - exp(-sigma_i * delta_i)
    #                     delta_i = t_i+1 - t_i
    #         data: dict of ground truth images
    #             'img_idxs': image indices; array of shape (N,) or (1,) if same image
    #             'pix_idxs': pixel indices; array of shape (N,)
    #             'pose': poses; array of shape (N, 3, 4)
    #             'direction': directions; array of shape (N, 3)
    #             'rgb': pixel colours; array of shape (N, 3)
    #             'depth': pixel depths; array of shape (N,)
    #         step: current training step; int
    #         loss: loss value; float
    #         color_loss: color loss value; float
    #         depth_loss: depth loss value; float
    #         tic: training starting time; time.time()
    #     """
    #     # calculate peak-signal-to-noise ratio
    #     with torch.no_grad():
    #         mse = F.mse_loss(results['rgb'], data['rgb'])
    #         psnr = -10.0 * torch.log(mse) / np.log(10.0)
    #         error, _, _, _, _, _ = self.evaluateDepth()

    #     # print statistics
    #     print(
    #         f"time={(time.time()-tic):.2f}s | "
    #         f"step={step} | "
    #         f"psnr={psnr:.2f} | "
    #         f"loss={loss:.4f} | "
    #         f"color_loss={color_loss:.4f} | "
    #         f"depth_loss={depth_loss:.4f} | "
    #         # number of rays
    #         f"rays={len(data['rgb'])} | "
    #         # ray marching samples per ray (occupied space on the ray)
    #         f"rm_s={results['rm_samples'] / len(data['rgb']):.1f} | "
    #         # volume rendering samples per ray 
    #         # (stops marching when transmittance drops below 1e-4)
    #         f"vr_s={results['vr_samples'] / len(data['rgb']):.1f} | "
    #         f"lr={(self.optimizer.param_groups[0]['lr']):.5f} | "
    #         f"depth_mae={error['depth_mae']:.3f} | "
    #         f"depth_mare={error['depth_mare']:.3f} | "
    #     )



        # def _depthLoss(self, results, data):
        # """
        # Loss function for training
        # Args:
        #     results: dict of rendered images
        #     data: dict of ground truth images
        # Returns:
        #     depth_loss: depth loss value; float
        # """
        # # val_idxs = ~torch.isnan(data['depth'])
        # # depth_loss = self.args.training.depth_loss_w * F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])
        # # if torch.all(torch.isnan(depth_loss)):
        # #     print("WARNING: trainer:lossFunc: depth_loss is nan, set to 0.")
        # #     depth_loss = 0

        # if self.args.rh.sensor_model == 'RGBD' or self.args.rh.sensor_model == 'ToF':
        #     val_idxs = ~torch.isnan(data['depth'])
        #     return F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])

        # if self.args.rh.sensor_model == 'USS':
        #     uss_mask = ~torch.isnan(data['depth'])
        #     # too_close = results['depth'] < data['depth']
        #     # if torch.any(too_close & uss_mask):
        #     #     depth_loss = F.mse_loss(results['depth'][too_close & uss_mask], data['depth'][too_close & uss_mask])
        #     # else:
        #     #     # depth_loss = (torch.min(results['depth']) - data['depth'][uss_mask][0])**2

        #     #     # conv_mask_size = 3
        #     #     # conv_mask = torch.ones(1,1,conv_mask_size,conv_mask_size).to(self.args.device)
        #     #     # W, H = self.train_dataset.img_wh
        #     #     # depth_results = torch.zeros(H*W).to(self.args.device) # (H*W)
        #     #     # depth_results[data['pix_idxs']] = results['depth']
        #     #     # depth_conv = F.conv2d(depth_results.reshape(H, W), weight=conv_mask, padding='same').reshape(H, W) # (H, W)
        #     #     # depth_conv = depth_conv.reshape(H*W)[data['pix_idxs']] # (N,)

        #     #     depths_w = torch.exp( -(results['depth'][uss_mask] - torch.min(results['depth'][uss_mask]))/0.1 )
        #     #     depths_w = depths_w / torch.sum(depths_w)
        #     #     depth_loss = torch.sum(depths_w * torch.abs(results['depth'][uss_mask] - data['depth'][uss_mask]))
        #     # return depth_loss


        #     # threshold = 0.2
        #     # depth_error = torch.abs(results['depth'] - data['depth'])
        #     # depth_mask = depth_error < threshold
        #     # return torch.mean( 0.5 * (1 - torch.cos(2*np.pi * depth_error[uss_mask & depth_mask] / threshold)) )

        #     threshold = 0.1

        #     depth_error = results['depth'][uss_mask] - data['depth'][uss_mask]
        #     cos_region_mask = (depth_error > -0.5*threshold) & (depth_error < threshold)
        #     lin_region_mask = depth_error <= -0.5*threshold

        #     losses = (2*threshold/np.pi) * torch.ones_like(depth_error).to(self.args.device)
        #     losses_cos = (threshold/np.pi) * (1 - torch.cos(2*np.pi * depth_error[cos_region_mask] / (2*threshold)).to(self.args.device))
        #     losses_lin = (2*threshold/np.pi) * (0.5 - np.pi/4 - depth_error[lin_region_mask] * np.pi / (2*threshold))
        #     losses[cos_region_mask] = losses_cos
        #     losses[lin_region_mask] = losses_lin
        #     return torch.mean(losses)
        
        # return 0.0