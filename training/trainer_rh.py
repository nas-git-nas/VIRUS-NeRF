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


# from gui import NGPGUI
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

class TrainerRH(Trainer):
    def __init__(self, hparams_file) -> None:
        Trainer.__init__(self, hparams_file=hparams_file)



    
    









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
    #         depth_tolerance = self.train_dataset.scene.w2c(pos=0.03, only_scale=True, copy=True)
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
    #     height_c = self.train_dataset.scene.w2c(pos=np.array([[0.0, 0.0, height_w]]), copy=True)[0,2]
    #     tolerance_c = self.train_dataset.scene.w2c(pos=tolerance_w, only_scale=True, copy=True)

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