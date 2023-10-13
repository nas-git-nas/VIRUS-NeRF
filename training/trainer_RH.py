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
import matplotlib.pyplot as plt

# from gui import NGPGUI
from datasets import dataset_dict
from datasets.ray_utils import get_rays

from modules.networks import NGP
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model
from helpers.geometric_fcts import findNearestNeighbour
from training.metrics_rh import MetricsRH



from training.trainer import Trainer

class TrainerRH(Trainer):
    def __init__(self, hparams_file) -> None:

        Trainer.__init__(self, hparams_file=hparams_file)

        # metrics
        self.metrics = MetricsRH(
            args=self.args,
            rh_scene=self.train_dataset.scene,
            img_wh=self.train_dataset.img_wh,
        )

    def train(self):
        """
        Training loop.
        """
        tic = time.time()
        for step in range(self.args.training.max_steps+1):
            self.model.train()

            i = torch.randint(0, len(self.train_dataset), (1,)).item()
            data = self.train_dataset[i]

            direction = data['direction']
            pose = data['pose']

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if step % self.args.occ_grid.update_interval == 0:
                    self.model.update_density_grid(
                        0.01 * MAX_SAMPLES / 3**0.5,
                        warmup=step < self.args.occ_grid.warmup_steps,
                    )
                # get rays and render image
                rays_o, rays_d = get_rays(direction, pose)
                results = render(
                    self.model, 
                    rays_o, 
                    rays_d,
                    exp_step_factor=self.args.exp_step_factor,
                )

                # calculate loss
                loss, color_loss, depth_loss = self.lossFunc(results=results, data=data)
                if self.args.training.distortion_loss_w > 0:
                    loss += self.args.training.distortion_loss_w * distortion_loss(results).mean()

            # backpropagate and update weights
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.scheduler.step()

            # TODO: print progress
            # if step % 100 == 0:
            #     self.__printStats(results=results, data=data, step=step, loss=loss, color_loss=color_loss, depth_loss=depth_loss, tic=tic)

        self.saveModel()

    def evaluate(self):
        """
        Evaluate NeRF on test set.
        """
        self.model.eval()

        # get indices of all test points and of one particular sensor
        img_idxs = np.arange(len(self.test_dataset))
        img_idxs_sensor = self.test_dataset.getIdxFromSensorName(df=self.test_dataset.df, sensor_name="RGBD_1")
        img_idxs_plot = np.copy(img_idxs_sensor)

        # keep only a certain number of points
        if self.args.eval.num_test_pts != "all":
            idxs_temp = np.linspace(0, len(img_idxs)-1, self.args.eval.num_test_pts, dtype=int)
            img_idxs = img_idxs[idxs_temp]

            idxs_temp = np.linspace(0, len(img_idxs_sensor)-1, self.args.eval.num_test_pts, dtype=int)
            img_idxs_sensor = img_idxs_sensor[idxs_temp]

        if self.args.eval.num_plot_pts != "all":
            idxs_temp = np.linspace(0, len(img_idxs_plot)-1, self.args.eval.num_plot_pts, dtype=int)
            img_idxs_plot = img_idxs_plot[idxs_temp]

        # evaluate color and depth
        color_metrics = self.evaluateColor(img_idxs=img_idxs)
        depth_metrics, data_w = self.evaluateDepth(img_idxs=img_idxs_sensor)
        metrics_dict = color_metrics.update(depth_metrics)

        print(
            f"evaluation: " \
            + f"psnr_avg={color_metrics['psnr']} | " \
            + f"ssim_avg={color_metrics['ssim']} | " \
            + f"depth_mae={depth_metrics['mae']} | " \
            + f"depth_mare={depth_metrics['mare']} | " \
            + f"depth_mnn={depth_metrics['mnn']} | " \
        )

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
        pix_idxs = np.arange(W*H)

        # get poses, direction and color ground truth
        poses = self.test_dataset.poses[img_idxs]
        directions = self.test_dataset.directions[pix_idxs]
        rgb_gt = self.test_dataset.rays[img_idxs, pix_idxs][:, :3]

        # calculate rays
        rays_o, rays_d = get_rays(
            directions=directions, 
            c2w=poses
        )

        # render rays to get color
        rgb = torch.empty(0, 3).to(self.args.device)
        depth = torch.empty(0).to(self.args.device)
        for results in self.__renderBatches(
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
        # create scan rays
        rays_o, rays_d = self.createScanRays(
            img_idxs=img_idxs,
            res_angular=self.args.eval.res_angular,
            num_avg_heights=self.args.eval.num_avg_heights,
            height_tolerance=self.args.eval.height_tolerance,
        ) # (N*M*A, 3), (N*M*A, 3)

        # render rays to get depth
        depths = torch.empty(0).to(self.args.device)
        for results in self.__renderBatches(
                rays_o=rays_o,
                rays_d=rays_d,
                test_time=True,
                batch_size=self.args.eval.batch_size,
            ):
            depths = torch.cat((depths, results['depth']), dim=0)

        # average dpeth over different heights
        depths = depths.detach().cpu().numpy().reshape(-1, self.args.eval.num_avg_heights) # (N*M, A)
        depth = np.nanmean(depths, axis=1) # (N*M,)

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
        rays_o_w = self.test_dataset.scene.c2wTransformation(pos=rays_o, copy=True) # (N*M*A, 3)
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
        )

        return metrics_dict, data_w
    

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
    
    def __renderBatches(
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
        if rays_o.shape[0] % batch_size == 0:
            num_batches = rays_o.shape[0] // batch_size
        else:
            num_batches = rays_o.shape[0] // batch_size + 1

        # render rays in batches
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i+1) * batch_size, rays_o.shape[0])
            results = render(
                self.model, 
                rays_o=rays_o[batch_start:batch_end], 
                rays_d=rays_d[batch_start:batch_end],
                test_time=test_time,
                exp_step_factor=self.args.exp_step_factor,
            )
            yield results
    
    def __printStats(self, results, data, step, loss, color_loss, depth_loss, tic):
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
            loss: loss value; float
            color_loss: color loss value; float
            depth_loss: depth loss value; float
            tic: training starting time; time.time()
        """
        # calculate peak-signal-to-noise ratio
        with torch.no_grad():
            mse = F.mse_loss(results['rgb'], data['rgb'])
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            error, _, _, _, _, _ = self.evaluateDepth()

        # print statistics
        print(
            f"time={(time.time()-tic):.2f}s | "
            f"step={step} | "
            f"psnr={psnr:.2f} | "
            f"loss={loss:.4f} | "
            f"color_loss={color_loss:.4f} | "
            f"depth_loss={depth_loss:.4f} | "
            # number of rays
            f"rays={len(data['rgb'])} | "
            # ray marching samples per ray (occupied space on the ray)
            f"rm_s={results['rm_samples'] / len(data['rgb']):.1f} | "
            # volume rendering samples per ray 
            # (stops marching when transmittance drops below 1e-4)
            f"vr_s={results['vr_samples'] / len(data['rgb']):.1f} | "
            f"lr={(self.optimizer.param_groups[0]['lr']):.5f} | "
            f"depth_mae={error['depth_mae']:.3f} | "
            f"depth_mare={error['depth_mare']:.3f} | "
        )

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
        colour_loss = self.__colorLoss(results=results, data=data)
        depth_loss = self.__depthLoss(results=results, data=data)
        
        depth_loss = depth_loss * self.args.training.depth_loss_w
        total_loss = colour_loss + depth_loss
        return total_loss, colour_loss, depth_loss


    def __colorLoss(self, results, data):
        """
        Loss function for training
        Args:
            results: dict of rendered images
            data: dict of ground truth images
        Returns:
            colour_loss: colour loss value; float
        """
        return F.mse_loss(results['rgb'], data['rgb'])
    
    def __depthLoss(self, results, data):
        """
        Loss function for training
        Args:
            results: dict of rendered images
            data: dict of ground truth images
        Returns:
            depth_loss: depth loss value; float
        """
        # val_idxs = ~torch.isnan(data['depth'])
        # depth_loss = self.args.training.depth_loss_w * F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])
        # if torch.all(torch.isnan(depth_loss)):
        #     print("WARNING: trainer:lossFunc: depth_loss is nan, set to 0.")
        #     depth_loss = 0

        if self.args.rh.sensor_model == 'USS':
            uss_mask = ~torch.isnan(data['depth'])
            # too_close = results['depth'] < data['depth']
            # if torch.any(too_close & uss_mask):
            #     depth_loss = F.mse_loss(results['depth'][too_close & uss_mask], data['depth'][too_close & uss_mask])
            # else:
            #     # depth_loss = (torch.min(results['depth']) - data['depth'][uss_mask][0])**2

            #     # conv_mask_size = 3
            #     # conv_mask = torch.ones(1,1,conv_mask_size,conv_mask_size).to(self.args.device)
            #     # W, H = self.train_dataset.img_wh
            #     # depth_results = torch.zeros(H*W).to(self.args.device) # (H*W)
            #     # depth_results[data['pix_idxs']] = results['depth']
            #     # depth_conv = F.conv2d(depth_results.reshape(H, W), weight=conv_mask, padding='same').reshape(H, W) # (H, W)
            #     # depth_conv = depth_conv.reshape(H*W)[data['pix_idxs']] # (N,)

            #     depths_w = torch.exp( -(results['depth'][uss_mask] - torch.min(results['depth'][uss_mask]))/0.1 )
            #     depths_w = depths_w / torch.sum(depths_w)
            #     depth_loss = torch.sum(depths_w * torch.abs(results['depth'][uss_mask] - data['depth'][uss_mask]))
            # return depth_loss


            # threshold = 0.2
            # depth_error = torch.abs(results['depth'] - data['depth'])
            # depth_mask = depth_error < threshold
            # return torch.mean( 0.5 * (1 - torch.cos(2*np.pi * depth_error[uss_mask & depth_mask] / threshold)) )

            threshold = 0.1

            depth_error = results['depth'][uss_mask] - data['depth'][uss_mask]
            cos_region_mask = (depth_error > -0.5*threshold) & (depth_error < threshold)
            lin_region_mask = depth_error <= -0.5*threshold

            losses = (2*threshold/np.pi) * torch.ones_like(depth_error).to(self.args.device)
            losses_cos = (threshold/np.pi) * (1 - torch.cos(2*np.pi * depth_error[cos_region_mask] / (2*threshold)).to(self.args.device))
            losses_lin = (2*threshold/np.pi) * (0.5 - np.pi/4 - depth_error[lin_region_mask] * np.pi / (2*threshold))
            losses[cos_region_mask] = losses_cos
            losses[lin_region_mask] = losses_lin
            return torch.mean(losses)

        if self.args.rh.sensor_model == 'ToF':
            val_idxs = ~torch.isnan(data['depth'])
            return F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])
        
        return 0.0



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