import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
import os
import sys
 
sys.path.insert(0, os.getcwd())
from datasets.sensor_model import ToFModel, USSModel
from datasets.dataset_rh import DatasetRH
from args.args import Args


def test_ToFModel():
    # hyperparameters
    num_imgs = 3

    # create dataset
    args = Args(
        file_name="rh_windows.json"
    )
    args.dataset.keep_N_observations = num_imgs
    args.training.sampling_strategy = {
        "imgs": "same",
        "rays": "entire_img",
    } 
    args.training.sensors = ["ToF", "RGBD"]
    dataset = DatasetRH(
        args = args,
        split="train",
    )
    W, H = dataset.img_wh

    # get depths
    depths_rgbd = np.zeros((num_imgs, W*H))
    depths_tof = np.zeros((num_imgs, W*H))
    for i, j in enumerate(np.linspace(0, len(dataset)-1, num_imgs, dtype=int)):
        data = dataset(
            batch_size=1,
            sampling_strategy=args.training.sampling_strategy,
            origin="nerf",
        )
        depths_rgbd[i] = data['depth']['RGBD'].detach().cpu().numpy()
        depths_tof[i] = data['depth']['ToF'].detach().cpu().numpy()

    # verify if ToF and RGBD depths are different
    valid_depth = ~np.isnan(depths_tof)
    same_depths = np.sum(np.abs(depths_rgbd[valid_depth] - depths_tof[valid_depth]) < 1e-2)
    print(f"ToF-RGBD Depths are at most 1cm apart per image: {(same_depths/num_imgs):.3} / {(np.sum(valid_depth)/num_imgs):.3}")

    # get masks for visualization
    mask = dataset.sensors_dict['ToF'].mask.astype(int)
    error_mask = dataset.sensors_dict['ToF'].error_mask.astype(int)
    mask_comb = np.zeros_like(mask, dtype=int)
    mask_comb[mask == 1] = 1
    mask_comb[error_mask == 1] = 2
    mask[mask == 1] = 1
    error_mask[error_mask == 1] = 2

    # plot
    fig, axes = plt.subplots(ncols=num_imgs, nrows=3, figsize=(12,8))
    depths_rgbd = depths_rgbd.reshape(num_imgs, H, W)
    depths_tof = depths_tof.reshape(num_imgs, H, W)
    mask = mask.reshape(H, W)
    error_mask = error_mask.reshape(H, W)
    mask_comb = mask_comb.reshape(H, W)

    # make single pixels visible
    depths_tof = skimage.measure.block_reduce(depths_tof, (1,8,8), np.nanmax) # (N, H, W)
    mask = skimage.measure.block_reduce(mask, (8,8), np.nanmax) # (H, W)
    error_mask = skimage.measure.block_reduce(error_mask, (8,8), np.nanmax) # (H, W)
    mask_comb = skimage.measure.block_reduce(mask_comb, (8,8), np.nanmax) # (H, W)

    for i in range(num_imgs):
        ax = axes[0,i]
        ax.imshow(depths_rgbd[i])
        ax.set_title(f'Depth Map GT: {i}')

        ax = axes[1,i]
        ax.imshow(depths_tof[i])
        ax.set_title(f'Depth Map ToF: {i}')

        if i == 0:
            ax = axes[2,i]
            ax.imshow(mask, vmin=0, vmax=2)
            ax.set_title(f'Mask (shift=0°)')
        if i == 1:
            ax = axes[2,i]
            ax.imshow(error_mask, vmin=0, vmax=2)
            ax.set_title(f'Error Mask (shift={args.tof.sensor_calibration_error}°)')
        if i == 2:
            ax = axes[2,i]
            ax.imshow(mask_comb, vmin=0, vmax=2)
            ax.set_title(f'Both Masks')
    
    plt.tight_layout()
    plt.show()


def test_USSModel():
    num_imgs = 3
    
    # img_wh = (64, 64)
    # depths = 100 * np.random.rand(num_imgs, img_wh[0]*img_wh[1])
    
    args = Args(file_name="rh_windows.json")
    args.rh.sensor_model = False
  
    dataset = DatasetRH(
        args = args,
        split="test",
    )
    # dataset.batch_size = 16
    dataset.pixel_sampling_strategy = 'entire_image'
    dataset.ray_sampling_strategy = 'same_image'
    img_wh = dataset.img_wh

    depths = np.zeros((num_imgs, img_wh[0]*img_wh[1]))
    for i, j in enumerate(np.linspace(0, len(dataset)-1, num_imgs, dtype=int)):
        data = dataset[j]
        depths[i] = data['depth'].detach().cpu().numpy()
        
    model = USSModel(args, img_wh)
    depths_out = model.convertDepth(depths)

    # plot
    fig, axes = plt.subplots(ncols=3, nrows=num_imgs, figsize=(12,8))
    depths = depths.reshape(num_imgs, img_wh[1], img_wh[0])
    depths_out = depths_out.reshape(num_imgs, img_wh[1], img_wh[0])
    mask = model.mask.reshape(img_wh[1], img_wh[0])

    # make single pixels visible
    depths_out = skimage.measure.block_reduce(depths_out, (1,4,4), np.nanmax) # (N, h, w)

    for i in range(depths.shape[0]):
        ax = axes[i,0]
        ax.imshow(depths[i], vmin=0, vmax=np.nanmax(depths))
        ax.set_title(f'Depth Map GT')

        ax = axes[i,1]
        ax.imshow(mask, vmin=0, vmax=1)
        ax.set_title(f'Mask')

        ax = axes[i,2]
        ax.imshow(depths_out[i], vmin=0, vmax=np.nanmax(depths))
        ax.set_title(f'Depth down sampled')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_ToFModel()
    # test_USSModel()



# def test_ComplexUSSModel():
#     num_imgs = 3
    
#     # img_wh = (8, 8)
#     # depths = 100 * np.random.rand(num_imgs, img_wh[0]*img_wh[1])
       
#     root_dir =  '../RobotAtHome2/data'   
#     dataset = RobotAtHomeDataset(
#         root_dir=root_dir,
#         split="test",
#     )
#     # dataset.batch_size = 16
#     dataset.pixel_sampling_strategy = 'entire_image'
#     dataset.ray_sampling_strategy = 'same_image'
#     img_wh = dataset.img_wh

#     depths = np.zeros((num_imgs, img_wh[0]*img_wh[1]))
#     for i, j in enumerate(np.linspace(0, len(dataset)-1, num_imgs, dtype=int)):
#         data = dataset[j]
#         depths[i] = data['depth'].detach().cpu().numpy()
        
#     model = USSModel(img_wh)
#     depths_out, depths_prob = model.convertDepth(depths, return_prob=True)

#     # plot
#     fig, axes = plt.subplots(ncols=4, nrows=num_imgs, figsize=(12,8))
#     depths = depths.reshape(num_imgs, img_wh[1], img_wh[0])
#     depths_out = depths_out.reshape(num_imgs, model.h, model.w)
#     depths_prob = depths_prob.reshape(num_imgs, model.h, model.w)
#     gaussian = model.gaussian.reshape(model.h, model.w)

#     for i in range(depths.shape[0]):
#         ax = axes[i,0]
#         ax.imshow(depths[i], vmin=0, vmax=np.nanmax(depths))
#         ax.set_title(f'Depth Map GT')

#         ax = axes[i,1]
#         ax.imshow(gaussian, vmin=0, vmax=1)
#         ax.set_title(f'Gaussian')

#         ax = axes[i,2]
#         ax.imshow(depths_prob[i], vmin=0, vmax=depths_prob[i].max())
#         ax.set_title(f'Depth Probability')

#         ax = axes[i,3]
#         ax.imshow(depths_out[i], vmin=0, vmax=np.nanmax(depths))
#         ax.set_title(f'Depth down sampled')
    
#     plt.tight_layout()
#     plt.show()