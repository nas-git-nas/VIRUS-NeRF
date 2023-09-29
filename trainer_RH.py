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
from datasets import dataset_dict
from datasets.ray_utils import get_rays

from modules.networks import NGP
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model

from torchmetrics import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
)

from trainer import Trainer

class TrainerRH(Trainer):
    def __init__(self) -> None:

        # TODO: add as hparams
        model_config = {
            'scale': 0.5,
            'pos_encoder_type': 'hash',
            'max_res':1024, #4096, # 1024,
            'half_opt': False,
        }

        Trainer.__init__(self, dataset=dataset_dict["robot_at_home"], model_config=model_config)

        # metric
        val_psnr = PeakSignalNoiseRatio(
            data_range=1
        ).to(self.args.device)
        val_ssim = StructuralSimilarityIndexMeasure(
            data_range=1
        ).to(self.args.device)

    def train(self):
        # training loop
        tic = time.time()
        for step in range(self.hparams.max_steps+1):
            self.model.train()

            i = torch.randint(0, len(self.train_dataset), (1,)).item()
            data = self.train_dataset[i]

            direction = data['direction']
            pose = data['pose']

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if step % self.hparams.update_interval == 0:
                    self.model.update_density_grid(
                        0.01 * MAX_SAMPLES / 3**0.5,
                        warmup=step < self.hparams.warmup_steps,
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
                loss = self.lossFunc(results=results, data=data)
                if self.hparams.distortion_loss_w > 0:
                    loss += self.hparams.distortion_loss_w * distortion_loss(results).mean()

            # backpropagate and update weights
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.scheduler.step()

            if step % 100 == 0:
                self.__printStats(results=results, data=data, step=step, loss=loss, tic=tic)

        self.__saveModel()

    def evaluate(self):
        # test loop
        progress_bar = tqdm.tqdm(total=len(self.test_dataset), desc=f'evaluating: ')
        with torch.no_grad():
            self.model.eval()
            w, h = self.test_dataset.img_wh
            directions = self.test_dataset.directions
            test_psnrs = []
            test_ssims = []
            for test_step in range(4): #range(len(test_dataset)): NS changed
                progress_bar.update()
                test_data = self.test_dataset[test_step]

                rgb_gt = test_data['rgb']
                poses = test_data['pose']

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # get rays
                    rays_o, rays_d = get_rays(directions, poses)
                    # render image
                    results = render(
                        self.model, 
                        rays_o, 
                        rays_d,
                        test_time=True,
                        exp_step_factor=self.args.exp_step_factor,
                    )


                # TODO: get rid of this
                rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
                rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
                # get psnr
                self.val_psnr(rgb_pred, rgb_gt)
                test_psnrs.append(self.val_psnr.compute())
                self.val_psnr.reset()
                # get ssim
                self.val_ssim(rgb_pred, rgb_gt)
                test_ssims.append(self.val_ssim.compute())
                self.val_ssim.reset()

                # save test image to disk
                if test_step == 0 or test_step == 10 or test_step == 100:
                    print(f"Saving test image {test_step} to disk")
                    test_idx = test_data['img_idxs']
                    # TODO: get rid of this
                    rgb_pred = rearrange(
                        results['rgb'].cpu().numpy(),
                        '(h w) c -> h w c',
                        h=h
                    )
                    rgb_pred = (rgb_pred * 255).astype(np.uint8)
                    depth = depth2img(
                        rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                    imageio.imsave(
                        os.path.join(
                            self.args.val_dir, 
                            f'rgb_{test_idx:03d}_'+str(test_step)+'.png'
                            ),
                        rgb_pred
                    )
                    imageio.imsave(
                        os.path.join(
                            self.args.val_dir, 
                            f'depth_{test_idx:03d}.png'
                        ),
                        depth
                    )

            progress_bar.close()
            test_psnr_avg = sum(test_psnrs) / len(test_psnrs)
            test_ssim_avg = sum(test_ssims) / len(test_ssims)
            with torch.no_grad():
                depth_mse = self.__evaluateDepth()
            print(f"evaluation: psnr_avg={test_psnr_avg} | ssim_avg={test_ssim_avg} | depth_mse={depth_mse}")


    def __printStats(self, results, data, step, loss, tic):
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
            tic: training starting time; time.time()
        """
        # calculate peak-signal-to-noise ratio
        with torch.no_grad():
            mse = F.mse_loss(results['rgb'], data['rgb'])
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            depth_mse = self.__evaluateDepth()

        # print statistics
        print(
            f"elapsed_time={(time.time()-tic):.2f}s | "
            f"step={step} | "
            f"psnr={psnr:.2f} | "
            f"loss={loss:.6f} | "
            # number of rays
            f"rays={len(data['rgb'])} | "
            # ray marching samples per ray (occupied space on the ray)
            f"rm_s={results['rm_samples'] / len(data['rgb']):.1f} | "
            # volume rendering samples per ray 
            # (stops marching when transmittance drops below 1e-4)
            f"vr_s={results['vr_samples'] / len(data['rgb']):.1f} | "
            f"lr={(self.optimizer.param_groups[0]['lr']):.5f} | "
            f"depth_mse={depth_mse:.6f} | "
        )

    def __evaluateDepth(self, res_angular=256):
        """
        Evaluate depth error.
        Args:
            res_angular: number of angular samples; int
        Returns:
            depth_mse: mean squared depth error; float
        """
        # get indices of one particular sensor
        sensor_img_idxs = self.test_dataset.getIdxFromSensorName(sensor_name="RGBD_1")
        sensor_img_idxs = torch.tensor(sensor_img_idxs, dtype=torch.long, device=self.args.device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # get rays
            rays_o = self.test_dataset.poses[sensor_img_idxs, :3, 3].detach().clone() # (N, 3)
            rays_o = torch.repeat_interleave(rays_o, res_angular, dim=0) # (N*M, 3)

            rays_d = torch.linspace(-np.pi, np.pi-2*np.pi/res_angular, res_angular, 
                                    dtype=torch.float32, device=self.args.device) # (M,)
            rays_d = torch.stack((torch.cos(rays_d), torch.sin(rays_d), torch.zeros_like(rays_d)), axis=1) # (M, 3)
            rays_d = rays_d.repeat(len(sensor_img_idxs), 1) # (N*M, 3)

            # render image
            results = render(
                self.model, 
                rays_o, 
                rays_d,
                test_time=True,
                exp_step_factor=self.args.exp_step_factor,
            )

        rays_o = rays_o.detach().cpu().numpy()
        rays_d = rays_d.detach().cpu().numpy()
        depth = results['depth'].detach().cpu().numpy() # (N*M,)

        # get ground truth depth
        scan_map, depth_gt, scan_angles = self.test_dataset.scene.getSliceScan(res=556, rays_o=rays_o, rays_d=rays_d, rays_o_in_world_coord=False)

        # convert depth to world coordinates (meters)
        depth = self.test_dataset.scene.c2wTransformation(pos=depth, only_scale=True, copy=False)
        depth_gt = self.test_dataset.scene.c2wTransformation(pos=depth_gt, only_scale=True, copy=False)

        # calculate mean squared depth error
        depth_mse = np.mean((depth-depth_gt)**2)

        return depth_mse





def test_trainer():
    trainer = TrainerRH()
    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    test_trainer()
