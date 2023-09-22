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

import matplotlib.pyplot as plt

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

warnings.filterwarnings("ignore")

def taichi_init(args):
    taichi_init_args = {"arch": ti.cuda,}
    if args.half_opt:
        taichi_init_args["half2_vectorization"] = True

    ti.init(**taichi_init_args)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hparams = get_opts()
    taichi_init(hparams)

    if hparams.deployment:
        model_config = {
            'scale': hparams.scale,
            'pos_encoder_type': 'hash',
            'levels': 4,
            'feature_per_level': 4,
            'base_res': 32,
            'max_res': 128,
            'log2_T': 21,
            'xyz_net_width': 16,
            'rgb_net_width': 16,
            'rgb_net_depth': 1,
        }
    else:
        model_config = {
            'scale': hparams.scale,
            'pos_encoder_type': hparams.encoder_type,
            'max_res': 1024 if hparams.scale == 0.5 else 4096,
            'half_opt': hparams.half_opt,
        }

    # model
    model = NGP(**model_config).to(device)

    # load checkpoint if ckpt path is provided
    hparams.ckpt_path = "results/Lego/model.pth"
    if hparams.ckpt_path:
        state_dict = torch.load(hparams.ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Load checkpoint from %s" % hparams.ckpt_path)

    # create slice
    slice_res = 128
    slice_heights = [-0.2, -0.1, 0.0, 0.1, 0.2]
    slice_pts = torch.linspace(-hparams.scale, hparams.scale, slice_res) # (slice_res,)
    m1, m2 = torch.meshgrid(slice_pts, slice_pts) # (slice_res,slice_res), (slice_res,slice_res)
    extent = [-hparams.scale,hparams.scale,-hparams.scale,hparams.scale]
       
    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=len(slice_heights), figsize=(12,6))
    
    for i in range(len(slice_heights)):
        # estimate density of slice
        height = slice_heights[i] * torch.ones(slice_res*slice_res,1)
        x = torch.cat((m1.reshape(-1,1), m2.reshape(-1,1), height), dim=1) # (slice_res*slice_res, 3)
        sigmas = model.density(x) # (slice_res*slice_res,3)
        sigmas = sigmas.reshape(slice_res, slice_res).cpu().detach().numpy() # (slice_res,slice_res)

        # threshold density
        threshold = 10
        sigmas_thresholded = sigmas.copy()
        sigmas_thresholded[sigmas < threshold] = 0.0
        sigmas_thresholded[sigmas >= threshold] = 1.0

        # Plot the density map for the current subplot
        ax = axes[0, i]
        ax.imshow(sigmas, extent=extent, origin='lower', cmap='viridis')
        ax.set_title(f'Height {slice_heights[i]}')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

        ax = axes[1, i]
        ax.imshow(sigmas_thresholded, extent=extent, origin='lower', cmap='viridis')
        ax.set_title(f'Thresholded at {threshold}')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
