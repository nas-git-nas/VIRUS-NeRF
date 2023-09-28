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

    # datasets
    root_dir ='../RobotAtHome2/data'
    dataset = dataset_dict["robot_at_home"]
    train_dataset = dataset(
        root_dir=root_dir,
        split="train",
        downsample=hparams.downsample,
    ).to(device)
    train_dataset.batch_size = hparams.batch_size
    train_dataset.ray_sampling_strategy = hparams.ray_sampling_strategy

    # model
    model = NGP(**model_config).to(device)

    # load checkpoint if ckpt path is provided
    hparams.ckpt_path = "results/rh_anto_livingroom1/model.pth"
    if hparams.ckpt_path:
        state_dict = torch.load(hparams.ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Load checkpoint from %s" % hparams.ckpt_path)

    # create slice
    slice_res = 128

    slice_heights_secene = [0.7045, 1.045, 1.345] # in scene coordinates (meters)
    pos = np.zeros((len(slice_heights_secene), 3))
    pos[:,2] = np.array(slice_heights_secene)
    pos = train_dataset.scalePosition(pos=pos)
    slice_heights = pos[:,2] # in cube coordinates [-0.5, 0.5]
    # slice_heights = [-0.2, -0.1, 0.0, 0.1, 0.2]

    # convert tolerance from meters to cube coordinates
    tolerance_scene = 0.1 # in meters
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, tolerance_scene]])
    pos = train_dataset.scalePosition(pos=pos)
    tolerance = pos[1,2] - pos[0,2] # in cube coordinates

    slice_pts = torch.linspace(-hparams.scale, hparams.scale, slice_res) # (slice_res,)
    m1, m2 = torch.meshgrid(slice_pts, slice_pts) # (slice_res,slice_res), (slice_res,slice_res)
    extent = [-hparams.scale,hparams.scale,-hparams.scale,hparams.scale]
       
    # Create a 3x3 grid of subplots
    thresholds = [5, 10, 15, 20]
    fig, axes = plt.subplots(ncols=2+len(thresholds), nrows=len(slice_heights), figsize=(12,6))
    
    for i in range(len(slice_heights)):

        # estimate density of slice
        density = []
        for j, height in enumerate(np.linspace(slice_heights[i]-tolerance, slice_heights[i]+tolerance, 10)):         
            x = torch.cat((m1.reshape(-1,1), m2.reshape(-1,1), height*torch.ones(slice_res*slice_res,1)), dim=1) # (slice_res*slice_res, 3)
            sigmas = model.density(x) # (slice_res*slice_res,3)
            sigmas = sigmas.reshape(slice_res, slice_res).cpu().detach().numpy() # (slice_res,slice_res)
            density.append(sigmas)
        density = np.array(density).mean(axis=0)
        

        # threshold density   
        density_thresholded = []
        for j, thr in enumerate(thresholds):
            density_thresholded.append(density.copy())
            density_thresholded[j][density < thr] = 0.0
            density_thresholded[j][density >= thr] = 1.0

        # get ground truth
        slice_map = train_dataset.getSceneSlice(height=slice_heights_secene[i], slice_res=slice_res, height_tolerance=tolerance_scene)

        # plot the ground truth
        ax = axes[i,0]
        ax.imshow(slice_map, extent=extent, origin='lower', cmap='viridis')
        if i == 0:
            ax.set_title(f'Ground Truth')
        ax.set_ylabel(f'Height {slice_heights_secene[i]}m')

        # Plot the density map for the current subplot
        ax = axes[i,1]
        ax.imshow(density, extent=extent, origin='lower', cmap='viridis')
        if i == 0:
            ax.set_title(f'Rendered Density')

        for j, sig_thr in enumerate(density_thresholded):
            ax = axes[i, j+2]
            ax.imshow(sig_thr, extent=extent, origin='lower', cmap='viridis')
            if i == 0:
                ax.set_title(f'Threshold = {thresholds[j]}')



    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
