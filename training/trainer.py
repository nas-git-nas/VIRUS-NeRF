import glob
import os
import time
import tqdm
import sys

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
import matplotlib.pyplot as plt

# from gui import NGPGUI
# from opt import get_opts
from args.args import Args
from datasets.ray_utils import get_rays
from datasets.dataset_base import DatasetBase

from modules.networks import NGP
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model

from modules.networks import NGP
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model
from helpers.geometric_fcts import findNearestNeighbour, createScanRays
from helpers.data_fcts import linInterpolateArray, convolveIgnorNans, dataConverged
from training.metrics_rh import MetricsRH

from modules.occupancy_grid import OccupancyGrid

from training.trainer_plot import TrainerPlot
from training.loss import Loss


from torchmetrics import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
)

warnings.filterwarnings("ignore")


class Trainer(TrainerPlot):
    def __init__(
        self, 
        hparams_file=None,
        args:Args=None,
        train_dataset:DatasetBase=None,
        test_dataset:DatasetBase=None,
    ) -> None:
        print(f"\n----- START INITIALIZING -----")

        TrainerPlot.__init__(
            self,
            args=args,
            hparams_file=hparams_file,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
        
        self.rng = np.random.default_rng(seed=self.args.seed)

        # # TODO: remove this
        # self.model.mark_invisible_cells(
        #     self.train_dataset.K,
        #     self.train_dataset.poses, 
        #     self.train_dataset.img_wh,
        # )

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

        # # scheduler
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=self.optimizer,
        #     T_max=self.args.training.max_steps,
        #     eta_min=self.args.training.lr/30,
        # )

        # loss function
        self.loss = Loss(
            args=self.args,
            scene=self.train_dataset.scene,
            sensors_dict=self.train_dataset.sensors_dict,
        )

        # metrics
        self.metrics = MetricsRH(
            args=self.args,
            scene=self.train_dataset.scene,
            img_wh=self.train_dataset.img_wh,
        )

        # initialize logs
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
                elapse_time=time.time()-train_tic,
            )
            
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
                            elapse_time=time.time()-train_tic,
                        )
                    else:
                        self.args.logger.error(f"grid_type {self.args.occ_grid.grid_type} not implemented")
                    

                # render image
                results = render(
                    self.model, 
                    data['rays_o'], 
                    data['rays_d'],
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
            # self.scheduler.step()

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
            

        self._saveModel()

    def evaluate(self):
        """
        Evaluate NeRF on test set.
        Returns:
            metrics_dict: dict of metrics; dict
        """
        print(f"\n----- START EVALUATING -----")
        self.model.eval()

        # get indices of all test points and of one particular sensor
        img_idxs = np.arange(len(self.test_dataset))
        img_idxs_sensor = self.test_dataset.getIdxFromSensorName( # TODO: add parameter or evaluation positions
            sensor_name="RGBD_1" if self.args.dataset.name == "RH2" else "CAM1",
        )

        # keep only a certain number of points
        if self.args.eval.num_color_pts != "all":
            idxs_temp = np.random.randint(0, len(img_idxs), self.args.eval.num_color_pts)
            # idxs_temp = np.linspace(0, len(img_idxs)-1, self.args.eval.num_color_pts, dtype=int)
            img_idxs = img_idxs[idxs_temp]

        if self.args.eval.num_depth_pts != "all":
            idxs_temp = np.linspace(0, len(img_idxs_sensor)-1, self.args.eval.num_depth_pts, dtype=int)
            img_idxs_sensor = img_idxs_sensor[idxs_temp]

        # evaluate color and depth
        color_dict = self._evaluateColor(img_idxs=img_idxs)
        metrics_dict, data_w = self._evaluateDepth(
            img_idxs=img_idxs_sensor,
            sensor_names=self.args.eval.sensors,
        )

        # create plots
        self._plotMetrics(
            metrics_dict=metrics_dict,
        )
        self._plotMaps(
            data_dict=data_w, 
            metrics_dict=metrics_dict,
            num_points=img_idxs_sensor.shape[0],
        )
        metrics_dict = self._plotLosses(
            logs=self.logs,
            metrics_dict=metrics_dict,
        )

        # print and save metrics
        for key in metrics_dict.keys():
            metrics_dict[key].update(color_dict)
        print(
            f"evaluation: " \
            + f"psnr_avg={np.round(metrics_dict['NeRF']['psnr'],2)} | " \
            + f"ssim_avg={metrics_dict['NeRF']['ssim']:.3} | " \
            + f"depth_mnn={metrics_dict['NeRF']['nn_mean']['zone3']:.3} | " \
        )

        metric_df = {key:[] for key in metrics_dict["NeRF"].keys()}
        metric_idxs = []
        for key in metrics_dict.keys():
            for metric, value in metrics_dict[key].items():
                metric_df[metric].append(value)
            metric_idxs.append(key)

        pd.DataFrame(
            data=metric_df,
            index=metric_idxs,
        ).to_csv(os.path.join(self.args.save_dir, "metrics.csv"), index=True)
        return metrics_dict

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
            # evaluate depth of one random image
            valid_img_idxs = self.test_dataset.sampler.getValidImgIdxs(
                elapse_time=time.time()-tic,
            )
            rand_ints = torch.randint(0, len(valid_img_idxs), (self.args.eval.num_depth_pts_per_step,), device=self.args.device)
            img_idxs = valid_img_idxs[rand_ints]
            depth_metrics, _ = self._evaluateDepth(
                img_idxs=img_idxs,
                sensor_names=["GT", "NeRF"],
            )

            # calculate peak-signal-to-noise ratio
            mse = F.mse_loss(results['rgb'], data['rgb'])
            psnr = -10.0 * torch.log(mse) / np.log(10.0)

            self.logs['psnr'][-1] = psnr.item()
            self.logs['mnn'][-1] = depth_metrics['NeRF']['nn_mean']['zone3']
            print(
                f"time={(time.time()-tic):.2f}s | "
                f"step={step} | "
                f"lr={(self.optimizer.param_groups[0]['lr']):.5f} | "
                f"loss={loss_dict['total']:.4f} | "
                f"color_loss={loss_dict['color']:.4f} | "
                f"depth_loss={loss_dict['depth']:.4f} | "
                f"psnr={psnr:.2f} | "
                f"depth_mnn={(depth_metrics['NeRF']['nn_mean']['zone3']):.3f} | "
            )  

    @torch.no_grad()
    def _evaluateColor(
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

        if N == 0:
            return {
                'psnr': -1.0,
                'ssim': -1.0,
            }

        # repeat image indices and pixel indices
        img_idxs = img_idxs.repeat(W*H) # (N*W*H,)
        pix_idxs = np.tile(np.arange(W*H), N) # (N*W*H,)

        # # get poses, direction and color ground truth
        # poses = self.test_dataset.poses[img_idxs]
        # directions_dict = self.test_dataset.directions_dict[pix_idxs]
        # rgb_gt = self.test_dataset.rgbs[img_idxs, pix_idxs][:, :3]

        # # calculate rays
        # rays_o, rays_d = get_rays(
        #     directions=directions, 
        #     c2w=poses
        # ) # (N*W*H, 3), (N*W*H, 3)

        data = self.test_dataset(
            img_idxs=torch.tensor(img_idxs, device=self.args.device),
            pix_idxs=torch.tensor(pix_idxs, device=self.args.device),
            elapse_time=1e12, # very large number -> use all data for evaluation
        )
        rays_o = data['rays_o']
        rays_d = data['rays_d']
        rgb_gt = data['rgb']

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
    def _evaluateDepth(
        self,
        img_idxs:np.array,
        sensor_names:list,
    ):
        """
        Sample points from NeRF and evaluate depth error.
        Args:
            img_idxs: image indices; array of int (N,)
            sensor_names: list of sensor names; list of str
        Returns:
            metrics_dict: dict of metrics
            data_w: dict of data in world coordinates
        """
        metrics_dict = {}
        data_dict = {}

        robot_pos, robot_orientation = self.test_dataset.getRobotPose2D(
            img_idxs=img_idxs,
            pose_in_world_coords=True,
        )
        data_dict["robot"] = {
            'pos':robot_pos,
            'orientation':robot_orientation,
        }

        fov, robot_pos, robot_orientation = self.test_dataset.getFieldOfView(
            img_idxs=img_idxs,
        )

        for sensor in sensor_names:
            # get data for evaluation
            rays_o, rays_d, depths = self._getEvaluationData(
                img_idxs=img_idxs,
                sensor=sensor,
            ) # (N*K, 3), (N*K, 3), (N*K,), (N, 2)

            # convert depth to positions: 3D -> 2D space
            pos, pos_o = self.test_dataset.scene.depth2pos(
                depths=depths,
                rays_o=rays_o,
                rays_d=rays_d,
            ) # (N*K, 2), (N*K, 2)

            if sensor == "GT":
                data_dict[sensor] = {
                    'pos': pos,
                    'pos_o': pos_o,
                    'depths': depths,
                    'rays_o': rays_o,
                }
                continue

            pos_gt, pos_o_gt = self._limitFoV(
                pos=data_dict["GT"]["pos"],
                pos_o=data_dict["GT"]["pos_o"],
                fov_sensor=fov[sensor],
                num_points=img_idxs.shape[0],
                robot_pos=robot_pos,
            ) # (N*M, 2), (N*M, 2)

            # calculate metrics
            nn_dists, nn_mean, nn_median, nn_inlier = self.metrics.nn(
                pos=pos,
                pos_ref=pos_gt,
                depths_gt=data_dict["GT"]["depths"],
                num_points=img_idxs.shape[0],
                ref_pos_is_gt=True,
            ) # (N*K,), (N*K,)

            nn_dists_inv, nn_mean_inv, nn_median_inv, nn_inlier_inv = self.metrics.nn(
                pos=pos_gt,
                pos_ref=pos,
                depths_gt=data_dict["GT"]["depths"],
                num_points=img_idxs.shape[0],
                ref_pos_is_gt=False,
            ) # (N*M,), (N*M,)

            data_dict[sensor] = {
                'pos': pos,
                'pos_o': pos_o,
                'depths': depths,
                'pos_gt': pos_gt,
                'pos_o_gt': pos_o_gt,
            }

            metrics_dict[sensor] = {
                'nn_dists': nn_dists,
                'nn_dists_inv': nn_dists_inv,
                'nn_mean': nn_mean,
                'nn_mean_inv': nn_mean_inv,
                'nn_inlier': nn_inlier,
                'nn_inlier_inv': nn_inlier_inv,
                'nn_median': nn_median,
                'nn_median_inv': nn_median_inv,
            }

        return metrics_dict, data_dict

    @torch.no_grad()      
    def _getEvaluationData(
        self,
        img_idxs:np.array,
        sensor:str,
    ):
        """
        Get evaluation data.
        Args:
            img_idxs: image indices; array of int (N,)
            sensor: sensor name; str
        Returns:
            rays_o: ray origins; array of shape (N*M, 3)
            rays_d: ray directions; array of shape (N*M, 3)
            depth: depth; array of shape (N*M,)
        """
        if sensor == "GT":
            return self._getEvaluationDataGT(
                img_idxs=img_idxs,
            )
        elif sensor == "NeRF":
            return self._getEvaluationDataNeRF(
                img_idxs=img_idxs,
            )
        elif sensor == "LiDAR":
            return self._getEvaluationDataLiDAR(
                img_idxs=img_idxs,
            )
        elif sensor == "ToF":
            return self._getEvaluationDataToFUSS(
                img_idxs=img_idxs,
                sensor_name="ToF",
            )
        elif sensor == "USS":
            return self._getEvaluationDataToFUSS(
                img_idxs=img_idxs,
                sensor_name="USS",
            )
        else:
            self.args.logger.error(f"sensor {sensor} not implemented")
            sys.exit()

    @torch.no_grad()
    def _getEvaluationDataGT(
        self,
        img_idxs:np.array,
    ):
        """
        Get evaluation data for ground truth.
        Args:
            img_idxs: image indices; array of int (N,)
        Returns:
            rays_o: ray origins; array of shape (N*M, 3)
            rays_d: ray directions; array of shape (N*M, 3)
            depth: depth; array of shape (N*M,)
        """
        # get ray origins
        rays_o_camera = self.test_dataset.poses[img_idxs, :3, 3].detach().clone().cpu().numpy() # (N, 3)
        rays_o = self.test_dataset.poses[img_idxs, :3, 3].detach().clone().cpu().numpy() # (N, 3)
        rays_o[:,2] = rays_o_camera[:,2]

        # create ray directions
        rays_o, rays_d = createScanRays(
            rays_o=rays_o,
            angle_res=self.args.eval.res_angular,
        ) # (N*M, 3), (N*M, 3)

        # determine depths
        _, depths, _ = self.test_dataset.scene.getSliceScan(
            res=self.args.eval.res_map, 
            rays_o=rays_o, 
            rays_d=rays_d, 
            rays_o_in_world_coord=False, 
            height_tolerance=self.args.eval.height_tolerance
        ) # (N*M,)
        
        # convert rays to world coordinates
        rays_o = self.test_dataset.scene.c2w(pos=rays_o, copy=False) # (N*M, 3)
        depths = self.test_dataset.scene.c2w(pos=depths, only_scale=True, copy=False) # (N*M,)
        return rays_o, rays_d, depths

    @torch.no_grad()
    def _getEvaluationDataNeRF(
        self,
        img_idxs:np.array,
    ):
        """
        Get evaluation data for NeRF.
        Args:
            img_idxs: image indices; array of int (N,)
        Returns:
            rays_o: ray origins; array of shape (N*M, 3)
            rays_d: ray directions; array of shape (N*M, 3)
            depth: depth; array of shape (N*M,)
        """
        # get ray origins
        rays_o_camera = self.test_dataset.poses[img_idxs, :3, 3].detach().clone() # (N, 3)
        rays_o = self.test_dataset.poses_lidar[img_idxs, :3, 3].detach().clone() # (N, 3)
        rays_o[:,2] = rays_o_camera[:,2]

        # create ray directions
        rays_o, rays_d = createScanRays(
            rays_o=rays_o,
            angle_res=self.args.eval.res_angular,
        ) # (N*M, 3), (N*M, 3)

        if self.args.model.debug_mode:
            # verify that all points are same
            rays_o_test = rays_o.detach().cpu().numpy().reshape(img_idxs.shape[0], -1, 3) # (N, M, 3)
            if not np.allclose(rays_o_test, rays_o_test[:,0,:][:,None,:]):
                self.args.logger.error(f"rays_o not all same...............")
                for i in range(img_idxs.shape[0]):
                    plt.scatter(rays_o_test[i,:,0], rays_o_test[i,:,1], c="b")
                    plt.scatter(rays_o_test[i,0,0], rays_o_test[i,0,1], c="r")
                    plt.show()
                sys.exit()

        # render rays to get depth
        depths = torch.empty(0).to(self.args.device) # (N*M,)
        for results in self._batchifyRender(
                rays_o=rays_o,
                rays_d=rays_d,
                test_time=True,
                batch_size=self.args.eval.batch_size,
            ):
            depths = torch.cat((depths, results['depth']), dim=0)

        # convert depth to world coordinates
        rays_o = rays_o.detach().clone().cpu().numpy() # (N*M, 3)
        rays_d = rays_d.detach().clone().cpu().numpy() # (N*M, 3)
        depths = depths.detach().clone().cpu().numpy()
        rays_o = self.test_dataset.scene.c2w(pos=rays_o, copy=False)
        depths = self.test_dataset.scene.c2w(pos=depths, only_scale=True, copy=False)

        if self.args.model.debug_mode:
            # verify that all points are same
            rays_o_test = rays_o.reshape(img_idxs.shape[0], -1, 3) # (N, M, 3)
            if not np.allclose(rays_o_test, rays_o_test[:,0,:][:,None,:]):
                self.args.logger.error(f"rays_o not all same")
                for i in range(img_idxs.shape[0]):
                    plt.scatter(rays_o_test[i,:,0], rays_o_test[i,:,1], c="b")
                    plt.scatter(rays_o_test[i,0,0], rays_o_test[i,0,1], c="r")
                    plt.show()
                sys.exit()

        return rays_o, rays_d, depths

    @torch.no_grad()
    def _getEvaluationDataLiDAR(
        self,
        img_idxs:np.array,
    ):
        """
        Get evaluation data for LiDAR.
        Args:
            img_idxs: image indices; array of int (N,)
        Returns:
            rays_o: ray origins; array of shape (N*M, 3)
            rays_d: ray directions; array of shape (N*M, 3)
            depth: depth; array of shape (N*M,)
            fov: field of view [min, max]; array of shape (N, 2)
        """
        xyzs, poses_lidar_w = self.test_dataset.getLidarMaps(
            img_idxs=img_idxs,
        )

        pos_cam_c = self.test_dataset.poses[img_idxs, :3, 3].detach().clone().cpu().numpy() # (N, 3)
        pos_cam_w = self.test_dataset.scene.c2w(pos=pos_cam_c, copy=True) # (N, 3)
        pos_lidar_w = poses_lidar_w[:, :3, 3] # (N, 3)

        # filter height slice from point cloud
        K = 0 # maximum number of points per point cloud
        for i, xyz in enumerate(xyzs):
            h_min = pos_cam_w[i,2] - self.args.eval.height_tolerance
            h_max = pos_cam_w[i,2] + self.args.eval.height_tolerance
            xyzs[i] = xyz[(xyz[:,2] >= h_min) & (xyz[:,2] <= h_max)] # (k, 3)

            if xyzs[i].shape[0] > K:
                K = xyzs[i].shape[0]

            # map_gt = self.test_dataset.scene.getSliceMap(
            #     height=pos_cam_w[i,2], 
            #     res=self.args.eval.res_map, 
            #     height_tolerance=self.args.eval.height_tolerance, 
            #     height_in_world_coord=True
            # )
            # scale = self.args.model.scale
            # extent = self.test_dataset.scene.c2w(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
            # extent = extent.T.flatten()
            # plt.imshow(map_gt.T, origin='lower', extent=extent)
            # plt.scatter(xyzs[i][:,0], xyzs[i][:,1], s=0.1, c="b")
            # plt.scatter(pos_cam_w[i,0], pos_cam_w[i,1], s=0.1, c="r")
            # plt.scatter(pos_lidar_w[i,0], pos_lidar_w[i,1], s=0.1, c="g")
            # plt.show()
        
        # determine rays
        depths = np.full((len(img_idxs), K), np.nan) # (N, K)
        rays_o = np.full((len(img_idxs), K, 3), np.nan) # (N, K, 3)
        rays_d = np.full((len(img_idxs), K, 3), np.nan) # (N, K, 3)
        for i, xyz in enumerate(xyzs):
            k = xyz.shape[0]
            pos_scan = np.concatenate((pos_lidar_w[i,:2].flatten(), pos_cam_w[i,2].flatten())) # (3,)
            rays_o[i, :k] = np.tile(pos_scan, (k, 1))
            rays_d[i, :k] = (xyz - pos_lidar_w[i]) / np.linalg.norm(xyz - pos_lidar_w[i], axis=1)[:,None]
            depths[i, :k] = np.linalg.norm(xyz - pos_lidar_w[i], axis=1)

        rays_o = rays_o.reshape(-1, 3) # (N*K, 3)
        rays_d = rays_d.reshape(-1, 3) # (N*K, 3)
        depths = depths.reshape(-1) # (N*K,)
            
        return rays_o, rays_d, depths

    @torch.no_grad()
    def _getEvaluationDataToFUSS(
        self,
        img_idxs:np.array,
        sensor_name:str,
    ):
        """
        Get evaluation data for ToF sensor.
        Args:
            img_idxs: image indices; array of int (N,)
            sensor_name: name of sensor, either "ToF" or "USS"; str
        Returns:
            rays_o: ray origins; array of shape (N*M, 3)
            rays_d: ray directions; array of shape (N*M, 3)
            depth: depth; array of shape (N*M,)
        """
        W, H = self.test_dataset.img_wh
        N = img_idxs.shape[0]
        img_idxs = torch.tensor(img_idxs, dtype=torch.int32, device=self.args.device) # (N,)

        # add synchrone samples from other sensor stack
        sync_idxs = self.test_dataset.getSyncIdxs(
            img_idxs=img_idxs,
        ) # (N, 2)
        img_idxs = sync_idxs.flatten() # (N*2,)

        # get pixel of viewing direction
        sensor_mask = self.test_dataset.sensors_dict[sensor_name].mask.detach().clone() # (H*W,)
        pix_idxs = torch.arange(H*W, dtype=torch.int32, device=self.args.device) # (H*W,)
        pix_idxs = pix_idxs[sensor_mask]

        # get positions, directions and depths of sensor
        img_idxs, pix_idxs = torch.meshgrid(img_idxs, pix_idxs, indexing="ij") # (N*2,M), (N*2,M)
        img_idxs = img_idxs.flatten() # (N*2*M,) = (N*k,)
        pix_idxs = pix_idxs.flatten() # (N*2*M,) = (N*k,)
        data = self.test_dataset(
            img_idxs=img_idxs,
            pix_idxs=pix_idxs,
            elapse_time=1e12, # very large number -> use all data for evaluation
        )

        rays_o = data['rays_o'].detach().cpu().numpy() # (N*k, 3)
        rays_d = data['rays_d'].detach().cpu().numpy() # (N*k, 3)
        depths = data['depth'][sensor_name].detach().cpu().numpy() # (N*k,)

        # convert rays to world coordinates
        rays_o = self.test_dataset.scene.c2w(pos=rays_o, copy=False) # (N*k, 3)
        depths = self.test_dataset.scene.c2w(pos=depths, only_scale=True, copy=False) # (N*k,)

        # filter rays using height tolerance
        mask = (depths * rays_d[:,2] >= -self.args.eval.height_tolerance) \
                & (depths * rays_d[:,2] <= self.args.eval.height_tolerance) \
                & (~np.isnan(depths)) # (N*k,)
        
        mask = mask.reshape(N, -1) # (N, k)
        rays_o = rays_o.reshape(N, -1, 3) # (N, k, 3)
        rays_d = rays_d.reshape(N, -1, 3) # (N, k, 3)  
        depths = depths.reshape(N, -1) # (N, k)
        K = np.max(np.sum(mask, axis=1)) # maximum number of points per point cloud

        rays_o_temp = np.full((N, K, 3), np.nan) # (N, K, 3)
        rays_d_temp = np.full((N, K, 3), np.nan) # (N, K, 3)
        depths_temp = np.full((N, K), np.nan) # (N, K)
        for i in range(N):
            k = np.sum(mask[i])
            rays_o_temp[i, :k, :] = rays_o[i, mask[i], :]
            rays_d_temp[i, :k, :] = rays_d[i, mask[i], :]
            depths_temp[i, :k] = depths[i, mask[i]]

        rays_o = rays_o_temp.reshape(-1, 3) # (N*K, 3)
        rays_d = rays_d_temp.reshape(-1, 3) # (N*K, 3)
        depths = depths_temp.reshape(-1) # (N*K,)

        if self.args.model.debug_mode:
            mask = np.all(~np.isnan(rays_o), axis=1)
            if not np.allclose(np.linalg.norm(rays_d[mask], axis=1), 1.0):
                self.args.logger.error(f"norm of rays_d is not 1.0: {np.linalg.norm(rays_d, axis=1)}")
                sys.exit()

        return rays_o, rays_d, depths
    
    def _sampleEvaluationData(
        self,
        pos:np.ndarray,
        pos_o:np.ndarray,
        num_points:int,
    ):
        """
        Sample evaluation data.
        Args:
            pos: positions; array of shape (N*K, 2)
            pos_o: positions of ray origins; array of shape (N*K, 2)
        Returns:
            pos: positions; array of shape (N*M, 2)
            pos_o: positions of ray origins; array of shape (N*M, 2)
        """
        N = num_points
        K = pos.shape[0] // N
        M = self.args.eval.res_angular

        # if K <= M:
        #     return rays_o, rays_d, depths

        pos = pos.reshape(N, K, 2) # (N, K, 2)
        pos_o = pos_o.reshape(N, K, 2) # (N, K, 2)
        angles = np.arctan2((pos - pos_o)[:,:,1], (pos - pos_o)[:,:,0]) # (N,K)
        dists = np.linalg.norm((pos - pos_o), axis=2) # (N,K)

        # set nan values to infinity to ignore them
        valid_mask = np.all(~np.isnan(pos), axis=2) # (N, K)
        angles[~valid_mask] = 0.0
        dists[~valid_mask] = np.inf

        # bin samples by angle
        angle_bins = np.linspace(-np.pi, np.pi, M) # (M+1,)
        angle_bin_idxs = np.digitize(angles, angle_bins) - 1 # (N,K), in range [0, M-1]
        if self.args.model.debug_mode:
            if np.max(angle_bin_idxs) >= M or np.min(angle_bin_idxs) < 0:
                self.args.logger.error(f"angle_bin_idxs out of range: max={np.max(angle_bin_idxs)}, min={np.min(angle_bin_idxs)}")
                self.args.logger.error(f"angles: max={np.max(angles)}, min={np.min(angles)}")
                self.args.logger.error(f"angle_bins: {angle_bins}")
                sys.exit()

        # project samples from measurement space (N,K) to angle bin space (N, M)
        mask = (angle_bin_idxs[:,:,None] == np.arange(M)[None,None,:]) # (N, K, M)
        dists = np.where(mask, dists[:,:,None], np.inf) # (N, K, M) 
        idxs = np.argmin(dists, axis=1) # (N, M) in range [0, K-1]
        pos = pos[np.arange(N).repeat(M), idxs.flatten()] # (N*M, 2)
        pos_o = pos_o[np.arange(N).repeat(M), idxs.flatten()] # (N*M, 2)

        # filter invalid samples
        valid_mask = (np.min(dists, axis=1) < np.inf) # (N, M)
        pos[~valid_mask.flatten()] = np.nan # (N*M, 2)
        pos_o[~valid_mask.flatten()] = np.nan # (N*M, 2)
        return pos, pos_o
    
    @torch.no_grad()
    def _limitFoV(
        self,
        fov_sensor:np.ndarray,
        pos:np.ndarray,
        pos_o:np.ndarray,
        num_points:int,
        robot_pos:dict,
    ):
        """
        Sample evaluation data for ground truth according to field of view of sensor.
        Args:
            fov_sensor: field of view [min, max]; dict {camera_name: array of shape (N, 2)}
            pos: positions; array of shape (N*M, 2)
            pos_o: positions of ray origins; array of shape (N*M, 2)
            num_points: number of points to sample; int
            robot_pos: robot positions; dict {camera_name: array of shape (N, 2)}
        Returns:
            pos: positions; array of shape (N*M, 2)
            pos_o: positions of ray origins; array of shape (N*M, 2)
        """
        # return pos, pos_o

        pos = pos.copy() # (N*M, 2)
        pos_o = pos_o.copy() # (N*M, 2)
        N = num_points 
        M = pos.shape[0] // N

        mask = np.zeros((N, M), dtype=bool) # (N, M)
        for name, fov in fov_sensor.items():
            # print(f"Sensor {name} has FoV")
            # for i in range(fov.shape[0]):
            #     print(f"     i={i}: {np.rad2deg(fov[i,0]):.2f} - {np.rad2deg(fov[i,1]):.2f}")

            if np.allclose(fov[:,0], -np.pi) and np.allclose(fov[:,1], np.pi):
                mask = np.ones((N, M), dtype=bool) # (N, M)
                break

            # calculate angles
            pos_o_temp = np.repeat(robot_pos[name], M, axis=0) # (N*M, 2)
            angles = np.arctan2((pos - pos_o_temp)[:,1], (pos - pos_o_temp)[:,0]) # (N*M,)
            angles = angles.reshape(N, M) # (N, M)

            angles_temp = angles - fov[:,0][:,None] # (N, M)
            upper_limit = fov[:,1] - fov[:,0] # (N,)
            angles_temp[angles_temp < 0] += 2*np.pi # (N, M)
            upper_limit[upper_limit < 0] += 2*np.pi # (N,)

            # add mask
            mask_temp = (angles_temp <= upper_limit[:,None]) # (N, M)
            mask = mask | mask_temp # (N, M)

        mask = mask.flatten() # (N*M,)
        pos[~mask] = np.nan # (N*M, 2)
        pos_o[~mask] = np.nan # (N*M, 2)

        return pos, pos_o






    # @torch.no_grad()
    # def _evaluateLidar(
    #         self, 
    #         img_idxs:np.array,
    # ) -> dict:
    #     """
    #     Evaluate depth error.
    #     Args:
    #         img_idxs: image indices; array of int (N,)
    #     Returns:
    #         metrics_dict: dict of metrics
    #         data_w: dict of data in world coordinates
    #     """
    #     xyzs, poses_lidar_w = self.test_dataset.getLidarMaps(
    #         img_idxs=img_idxs,
    #     )

    #     pos_cam_c = self.test_dataset.poses[img_idxs, :3, 3].detach().clone().cpu().numpy() # (N, 3)
    #     pos_cam_w = self.test_dataset.scene.c2w(pos=pos_cam_c, copy=True) # (N, 3)

    #     depths = np.zeros((len(img_idxs), self.args.eval.res_angular)) # (N, M)
    #     rays_o = np.zeros((len(img_idxs), self.args.eval.res_angular, 3)) # (N, M, 3)
    #     rays_d = np.zeros((len(img_idxs), self.args.eval.res_angular, 3)) # (N, M, 3)
    #     for i, xyz in enumerate(xyzs):
    #         pos_lidar_w = poses_lidar_w[i, :3, 3] # (3,)
            
    #         # filter height slice from point cloud
    #         h_min = pos_cam_w[i,2] - self.args.eval.height_tolerance #- self.args.lidar.height_offset
    #         h_max = pos_cam_w[i,2] + self.args.eval.height_tolerance #- self.args.lidar.height_offset
    #         xyz_n = xyz[(xyz[:,2] >= h_min) & (xyz[:,2] <= h_max)] # (n, 3)
 
    #         # sample fixed number of points
    #         pos_rel_n = xyz_n - pos_lidar_w[None,:] # (n, 3)
    #         beta_n = np.arctan2(pos_rel_n[:,1], pos_rel_n[:,0]) # (n,)
    #         bins = np.linspace(-np.pi, np.pi, self.args.eval.res_angular+1) # (M+1,)
    #         bin_idxs = np.digitize(beta_n, bins) - 1 # (n,), in range [0, M-1]
            
    #         depth_mn = np.full((self.args.eval.res_angular, bin_idxs.shape[0]), np.nan) # (M, n)
    #         depth_mn[bin_idxs, np.arange(bin_idxs.shape[0])] = np.linalg.norm(pos_rel_n, axis=1) # (M, n)
    #         sort_idxs = np.argsort(depth_mn, axis=1) # (M, n)
    #         num_elements = np.sum(~np.isnan(depth_mn), axis=1) # (M,)
    #         median_idxs = sort_idxs[np.arange(self.args.eval.res_angular), num_elements//2] # (M,) in range [0, n-1]

    #         depth_m = depth_mn[np.arange(self.args.eval.res_angular), median_idxs] # (M,)
    #         pos_rel_m = pos_rel_n[median_idxs] # (M, 3)

    #         # consider viewing direction in xy-plane to be equally spaced, but in z-direction to be proportional to sample
    #         alpha_m = np.arctan2(pos_rel_m[:,2], np.sqrt(pos_rel_m[:,1]**2 + pos_rel_m[:,0]**2)) # (M,)
    #         beta_m = np.linspace(
    #             start=-np.pi + np.pi/self.args.eval.res_angular, 
    #             stop=np.pi - np.pi/self.args.eval.res_angular, 
    #             num=self.args.eval.res_angular,
    #         ) # (M,)
    #         ray_d_m = np.vstack((np.cos(beta_m)*np.cos(alpha_m), np.sin(beta_m)*np.cos(alpha_m), np.sin(alpha_m))).T # (M, 3)

    #         # project depth on xy-plane
    #         depth_m = depth_m * np.cos(alpha_m)

    #         # save data
    #         depths[i] = depth_m
    #         rays_o[i] = np.tile(pos_lidar_w, (self.args.eval.res_angular, 1)) # (M, 3)
    #         rays_d[i] = ray_d_m
            
    #         if self.args.model.debug_mode:
    #             if not np.allclose(np.linalg.norm(ray_d_m, axis=1), 1.0):
    #                 self.args.logger.error(f"----lidar rays not normalized: {np.linalg.norm(rays_d[i], axis=1).max()}")

    #     # convert lidar height to camera height
    #     rays_o[:,:,2] = pos_cam_w[:,2][:,None] # (N, M, 3)

    #     # convert depth an rays to cube coordinates
    #     depths_c = self.test_dataset.scene.w2c(pos=depths.reshape(-1), only_scale=True, copy=True) # (N*M,)
    #     rays_o_c = self.test_dataset.scene.w2c(pos=rays_o.reshape(-1, 3), copy=True) # (N*M, 3)
    #     rays_d_c = rays_d.reshape(-1, 3) # (N*M, 3)

    #     metrics_dict, data_w = self._evaluateDepthMetric(
    #         rays_o=rays_o_c, # (N*M, 3)
    #         rays_d=rays_d_c, # (N*M, 3)
    #         depth=depths_c, # (N*M,)
    #         num_test_pts=len(img_idxs),
    #     )

    #     N = len(img_idxs)
    #     d_depth = data_w['depth'].reshape(N, self.args.eval.res_angular) # (N, M)
    #     d_depth_gt = data_w['depth_gt'].reshape(N, self.args.eval.res_angular) # (N, M)
    #     d_rays_o = data_w['rays_o'].reshape(N, self.args.eval.res_angular, 3) # (N, M, 3)
    #     d_scan_angles = data_w['scan_angles'].reshape(N, self.args.eval.res_angular) # (N, M)
    #     d_scan_map_gt = data_w['scan_map_gt'] # (L, L)

    #     scale = self.args.model.scale
    #     extent = self.test_dataset.scene.c2w(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
    #     extent = extent.T.flatten()

    #     for i in range(N):

    #         fig, axs = plt.subplots(2, 2, figsize=(10,5))

    #         # plot lidar map
    #         ax = axs[0,0]
    #         ax.imshow(d_scan_map_gt.swapaxes(0,1), origin='lower', extent=extent)
    #         ax.scatter(positions_w_cam[i,0], positions_w_cam[i,1], c='b')
    #         ax.scatter(rays_o[i,0,0], rays_o[i,0,1], c='r')
    #         xs = [rays_o[i,:,0], rays_o[i,:,0]+rays_d[i,:,0]*depths[i]]
    #         ys = [rays_o[i,:,1], rays_o[i,:,1]+rays_d[i,:,1]*depths[i]]
    #         ax.plot(xs, ys, c='w', alpha=0.4)
    #         ax.set_title(f"data in")
    #         ax.set_xlim(extent[0], extent[1])
    #         ax.set_ylim(extent[2], extent[3])

    #         ax = axs[0,1]
    #         ax.imshow(d_scan_map_gt.swapaxes(0,1), origin='lower', extent=extent)
    #         ax.scatter(d_rays_o[i,0,0], d_rays_o[i,0,1], c='r')
    #         xs3 = [d_rays_o[i,:,0], d_depth[i]*np.cos(d_scan_angles[i,:])+d_rays_o[i,:,0]]
    #         ys3 = [d_rays_o[i,:,1], d_depth[i]*np.sin(d_scan_angles[i,:])+d_rays_o[i,:,1]]
    #         ax.plot(xs3, ys3, c='w', alpha=0.4)
    #         ax.set_title(f"data out")
    #         ax.set_xlim(extent[0], extent[1])
    #         ax.set_ylim(extent[2], extent[3])


    #         ax = axs[1,0]
    #         ax.imshow(d_scan_map_gt.swapaxes(0,1), origin='lower', extent=extent)
    #         ax.scatter(d_rays_o[i,:,0], d_rays_o[i,:,1], c='r')
    #         xs = [d_rays_o[i,:,0], d_depth_gt[i,:]*np.cos(d_scan_angles[i,:])+d_rays_o[i,:,0]]
    #         ys = [d_rays_o[i,:,1], d_depth_gt[i,:]*np.sin(d_scan_angles[i,:])+d_rays_o[i,:,1]]
    #         ax.plot(xs, ys, c='w', alpha=0.4)
    #         ax.set_title(f"gt")
    #         ax.set_xlim(extent[0], extent[1])
    #         ax.set_ylim(extent[2], extent[3])

    #         ax = axs[1,1]
    #         ax.imshow(d_scan_map_gt.swapaxes(0,1), origin='lower', extent=extent)
    #         ax.set_xlim(extent[0], extent[1])
    #         ax.set_ylim(extent[2], extent[3])

    #         plt.show()

    #     return metrics_dict, data_w

    # def _evaluateDepth(
    #         self, 
    #         img_idxs:np.array,
    #         return_only_nerf:bool=False,
    # ):
    #     """
    #     Evaluate depth error.
    #     Args:
    #         img_idxs: image indices; array of int (N,)
    #         return_only_nerf: return only metrics for NeRF; bool
    #     Returns:
    #         metrics_dict: dict of metrics
    #         data_w: dict of data in world coordinates
    #     """
    #     metrics_dict_nerf, data_w_nerf = self._evaluateDepthNeRF(
    #         img_idxs=img_idxs,
    #     )
    #     if return_only_nerf:
    #         return metrics_dict_nerf, data_w_nerf
        
    #     metrics_dict_lidar, data_w_lidar = self._evaluateLidar(
    #         img_idxs=img_idxs,
    #     )

    #     metrics_dict_tof, data_w_tof = self._evaluateDepthSensor(
    #         img_idxs=img_idxs,
    #         sensor_name="ToF",
    #     )

    #     metrics_dict_uss, data_w_uss = self._evaluateDepthSensor(
    #         img_idxs=img_idxs,
    #         sensor_name="USS",
    #     )

    #     metrics_dict = {
    #         "NeRF": metrics_dict_nerf,
    #         "ToF": metrics_dict_tof,
    #         "USS": metrics_dict_uss,
    #         "LiDAR": metrics_dict_lidar,
    #     }

    #     data_w = { 
    #         'rays_o_nerf': data_w_nerf['rays_o'],
    #         'rays_o_tof': data_w_tof['rays_o'],
    #         'rays_o_uss': data_w_uss['rays_o'],
    #         'rays_o_lidar': data_w_lidar['rays_o'],
    #         'scan_map_gt': data_w_nerf['scan_map_gt'],
    #         'depth_gt_nerf': data_w_nerf['depth_gt'],
    #         'depth_gt_tof': data_w_tof['depth_gt'],
    #         'depth_gt_uss': data_w_uss['depth_gt'],
    #         'depth_gt_lidar': data_w_lidar['depth_gt'],
    #         'depth_nerf': data_w_nerf['depth'],
    #         'depth_tof': data_w_tof['depth'],
    #         'depth_uss': data_w_uss['depth'],
    #         'depth_lidar': data_w_lidar['depth'],
    #         'scan_angles_nerf': data_w_nerf['scan_angles'],
    #         'scan_angles_tof': data_w_tof['scan_angles'],
    #         'scan_angles_uss': data_w_uss['scan_angles'],
    #         'scan_angles_lidar': data_w_lidar['scan_angles'],
    #     }
    #     return metrics_dict, data_w

    # @torch.no_grad()
    # def _evaluateDepthNeRF(
    #     self, 
    #     img_idxs:np.array,
    # ) -> dict:
    #     """
    #     Evaluate depth error.
    #     Args:
    #         img_idxs: image indices; array of int (N,)
    #     Returns:
    #         metrics_dict: dict of metrics
    #         data_w: dict of data in world coordinates
    #     """
    #     # use xy-positions of lidar, but height of camera
    #     pos_cam_c = self.test_dataset.poses[img_idxs, :3, 3].detach().clone() # (N, 3)
    #     pos_lidar_c = self.test_dataset.poses_lidar[img_idxs, :3, 3].detach().clone() # (N, 3)
    #     pos_lidar_c[:,2] = pos_cam_c[:,2] # (N, 3)

    #     # convert height tolerance to cube coordinates
    #     h_tol_c = self.test_dataset.scene.w2c(pos=self.args.eval.height_tolerance, only_scale=True, copy=True)

    #     # create scan rays for averaging over different heights
    #     rays_o, rays_d = createScanRays(
    #         rays_o=pos_lidar_c,
    #         res_angular=self.args.eval.res_angular,
    #         h_tol_c=h_tol_c,
    #         num_avg_heights=self.args.eval.num_avg_heights,
    #     ) # (N*M*A, 3), (N*M*A, 3)

    #     # render rays to get depth
    #     depths = torch.empty(0).to(self.args.device)
    #     for results in self._batchifyRender(
    #             rays_o=rays_o,
    #             rays_d=rays_d,
    #             test_time=True,
    #             batch_size=self.args.eval.batch_size,
    #         ):
    #         depths = torch.cat((depths, results['depth']), dim=0)

    #     # average dpeth over different heights
    #     depths = depths.detach().cpu().numpy().reshape(-1, self.args.eval.num_avg_heights) # (N*M, A)
    #     depth = np.nanmean(depths, axis=1) # (N*M,)


    #     # create scan rays for averaging over different heights
    #     rays_o, rays_d = createScanRays(
    #         rays_o=pos_lidar_c,
    #         res_angular=self.args.eval.res_angular,
    #         h_tol_c=0.0,
    #         num_avg_heights=1,
    #     ) # (N*M*A, 3), (N*M*A, 3)

    #     metrics_dict, data_w = self._evaluateDepthMetric(
    #         rays_o=rays_o.detach().cpu().numpy(), # (N*M, 3), 
    #         rays_d=rays_d.detach().cpu().numpy(), # (N*M, 3), 
    #         depth=depth, 
    #         num_test_pts=len(img_idxs),
    #     )
    #     return metrics_dict, data_w
    
    # @torch.no_grad()
    # def _evaluateDepthSensor(
    #         self, 
    #         img_idxs:np.array,
    #         sensor_name:str,
    # ) -> dict:
    #     """
    #     Evaluate depth error.
    #     Args:
    #         img_idxs: image indices; array of int (N,)
    #         sensor_name: name of sensor; str
    #     Returns:
    #         metrics_dict: dict of metrics
    #         data_w: dict of data in world coordinates
    #     """
    #     img_idxs = torch.tensor(img_idxs, dtype=torch.int32, device=self.args.device) # (N,)
    #     W, H = self.test_dataset.img_wh

    #     # add synchrone samples from other sensor stack
    #     sync_idxs = self.test_dataset.getSyncIdxs(
    #         img_idxs=img_idxs,
    #     )
    #     img_idxs = sync_idxs.flatten() # N->2*N

    #     # determine scan pixel height
    #     sensor_mask = self.test_dataset.sensors_dict[sensor_name].mask.detach().clone().reshape(H,W) # (H, W)
    #     if sensor_name == "USS":
    #         scan_pix_h = H//2
    #     elif sensor_name == "ToF":
    #         sensor_mask_height = (torch.sum(sensor_mask, dim=1) > 0)
    #         scan_pix_h = torch.arange(H, device=self.args.device)[sensor_mask_height][3]
    #     else:
    #         self.args.logger.error(f"sensor_name {sensor_name} not implemented")

    #     # get pixel indices of sensor
    #     scan_mask = torch.zeros_like(sensor_mask, dtype=torch.bool, device=self.args.device) # (H, W)
    #     scan_mask[scan_pix_h, :] = sensor_mask[scan_pix_h, :] # (H, W)
    #     scan_mask = scan_mask.reshape(-1) # (H*W,)
    #     pix_idxs = torch.arange(H*W, dtype=torch.int32, device=self.args.device) # (H*W,)
    #     pix_idxs = pix_idxs[scan_mask]

    #     # get positions, directions and depths of sensor
    #     img_idxs, pix_idxs = torch.meshgrid(img_idxs, pix_idxs, indexing="ij") # (2*N,M), (2*N,M)
    #     data = self.test_dataset(
    #         img_idxs=img_idxs.flatten(),
    #         pix_idxs=pix_idxs.flatten(),
    #         elapse_time=1e12, # very large number -> use all data for evaluation
    #     )

    #     rays_o = data['rays_o'].detach().cpu().numpy() # (2*N*M, 3)
    #     rays_d = data['rays_d'].detach().cpu().numpy() # (2*N*M, 3)
    #     depth = data['depth'][sensor_name].detach().cpu().numpy() # (2*N*M,)

    #     # project depth on xy-plane
    #     alpha = np.arctan2(rays_d[:,2], np.sqrt(rays_d[:,0]**2+rays_d[:,1]**2))
    #     depth = depth * np.cos(alpha)

    #     metrics_dict, data_w = self._evaluateDepthMetric(
    #         rays_o=rays_o, # (N*M, 3),
    #         rays_d=rays_d, # (N*M, 3),
    #         depth=depth, # (N*M,)
    #         num_test_pts=len(img_idxs),
    #     )
    #     return metrics_dict, data_w
    
   

    # @torch.no_grad()
    # def _evaluateDepthMetric(
    #         self, 
    #         rays_o:np.array,
    #         rays_d:np.array,
    #         depth:np.array,
    #         num_test_pts:int,
    # ) -> dict:
    #     """
    #     Evaluate depth error. The depth is assumed to be projected onto the xy-plane.
    #     Args:
    #         rays_o: ray origins; array of shape (N*M, 3)
    #         rays_d: ray directions; array of shape (N*M, 3)
    #         depth: depth; array of shape (N*M,)
    #         num_test_pts: number of test points N; int
    #     Returns:
    #         metrics_dict: dict of metrics
    #         data_w: dict of data in world coordinates
    #     """
    #     # get ground truth depth
    #     scan_map_gt, depth_gt, scan_angles = self.test_dataset.scene.getSliceScan(
    #         res=self.args.eval.res_map, 
    #         rays_o=rays_o, 
    #         rays_d=rays_d, 
    #         rays_o_in_world_coord=False, 
    #         height_tolerance=self.args.eval.height_tolerance
    #     )

    #     # convert depth to world coordinates (meters)
    #     depth_w = self.test_dataset.scene.c2w(pos=depth, only_scale=True, copy=True)
    #     depth_w_gt = self.test_dataset.scene.c2w(pos=depth_gt, only_scale=True, copy=True)
    #     rays_o_w = self.test_dataset.scene.c2w(pos=rays_o, copy=True) # (N*M, 3)
    #     data_w = {
    #         'depth': depth_w,
    #         'depth_gt': depth_w_gt,
    #         'rays_o': rays_o_w,
    #         'scan_angles': scan_angles,
    #         'scan_map_gt': scan_map_gt,
    #     }

    #     # calculate mean squared depth error
    #     metrics_dict = self.metrics.evaluate(
    #         data=data_w,
    #         eval_metrics=['rmse', 'mae', 'mare', 'nn', 'nn_inv', 'rnn', 'rnn_inv'],
    #         convert_to_world_coords=False,
    #         copy=True,
    #         num_test_pts=num_test_pts,
    #     )
    #     return metrics_dict, data_w


    
    

    
    




    


