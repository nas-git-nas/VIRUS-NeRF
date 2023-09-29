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
                # get rays
                rays_o, rays_d = get_rays(direction, pose)
                # render image
                results = render(
                    self.model, 
                    rays_o, 
                    rays_d,
                    exp_step_factor=self.args.exp_step_factor,
                )
                loss = self.lossFunc(results=results, data=data)
                if self.hparams.distortion_loss_w > 0:
                    loss += self.hparams.distortion_loss_w * distortion_loss(results).mean()

            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.scheduler.step()

            if step % 100 == 0:
                elapsed_time = time.time() - tic
                with torch.no_grad():
                    mse = F.mse_loss(results['rgb'], data['rgb'])
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                print(
                    f"elapsed_time={elapsed_time:.2f}s | "
                    f"step={step} | psnr={psnr:.2f} | "
                    f"loss={loss:.6f} | "
                    # number of rays
                    f"rays={len(data['rgb'])} | "
                    # ray marching samples per ray (occupied space on the ray)
                    f"rm_s={results['rm_samples'] / len(data['rgb']):.1f} | "
                    # volume rendering samples per ray 
                    # (stops marching when transmittance drops below 1e-4)
                    f"vr_s={results['vr_samples'] / len(data['rgb']):.1f} | "
                    f"lr={(self.optimizer.param_groups[0]['lr']):.5f} | "
                )


        # save model
        print(f"Saving model to {self.val_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.val_dir, 'model.pth'),
        )

    def test(self):
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
                            self.args.__annotations__val_dir, 
                            f'depth_{test_idx:03d}.png'
                        ),
                        depth
                    )

            progress_bar.close()
            test_psnr_avg = sum(test_psnrs) / len(test_psnrs)
            test_ssim_avg = sum(test_ssims) / len(test_ssims)
            print(f"evaluation: psnr_avg={test_psnr_avg} | ssim_avg={test_ssim_avg}")




def test_trainer():
    trainer = TrainerRH()
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    test_trainer()
