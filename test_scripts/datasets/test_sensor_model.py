import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
import os
import sys
from scipy.ndimage import grey_dilation
 
sys.path.insert(0, os.getcwd())
from datasets.dataset_rh import DatasetRH
from datasets.dataset_ethz import DatasetETHZ
from args.args import Args


def test_ToFModel():
    # hyperparameters
    num_imgs = 5

    # create dataset
    args = Args(
        file_name="ethz_usstof_win.json"
    )
    args.dataset.keep_N_observations = num_imgs
    args.training.sampling_strategy = {
        "imgs": "same",
        "pixs": "entire_img",
    } 
    args.training.sensors = ["ToF", "RGBD"]
    args.dataset.sensors = ["ToF", "RGBD"]
    args.tof.tof_pix_size = 32

    if args.dataset.name == "RH2":
        dataset_class = DatasetRH
    elif args.dataset.name == "ETHZ":
        dataset_class = DatasetETHZ
    dataset = dataset_class(
        args = args,
        split="train",
    )
    W, H = dataset.img_wh

    # get depths
    depths_rgbd = np.zeros((num_imgs, W*H))
    depths_tof = np.zeros((num_imgs, W*H))
    for i, j in enumerate(np.linspace(0, len(dataset)-1, num_imgs, dtype=int)):
        data = dataset(
            batch_size=W*H,
            sampling_strategy=args.training.sampling_strategy,
        )
        depths_rgbd[i] = data['depth']['RGBD'].detach().cpu().numpy()
        depths_tof[i] = data['depth']['ToF'].detach().cpu().numpy()

    # verify if ToF and RGBD depths are different
    valid_depth_tof = ~np.isnan(depths_tof)
    valid_depth_rgbd = ~np.isnan(depths_rgbd)
    valid_depth = valid_depth_tof & valid_depth_rgbd
    same_depths = np.sum(np.abs(depths_rgbd[valid_depth] - depths_tof[valid_depth]) < 0.05)
    print(f"ToF-RGBD Depths are at most 5cm apart per image: {same_depths} / {np.sum(valid_depth)} = {(same_depths/np.sum(valid_depth)):.3}%")

    # get masks for visualization
    mask = dataset.sensors_dict['ToF'].mask.cpu().numpy().astype(int)
    error_mask = dataset.sensors_dict['ToF'].error_mask.cpu().numpy().astype(int)
    mask_comb = np.zeros_like(mask, dtype=int)
    mask_comb[mask == 1] = 1
    mask_comb[error_mask == 1] = 2
    mask[mask == 1] = 1
    error_mask[error_mask == 1] = 2

    # plot
    fig, axes = plt.subplots(ncols=num_imgs, nrows=3, figsize=(15,8))
    depths_rgbd = depths_rgbd.reshape(num_imgs, H, W)
    depths_tof = depths_tof.reshape(num_imgs, H, W)
    mask = mask.reshape(H, W)
    error_mask = error_mask.reshape(H, W)
    mask_comb = mask_comb.reshape(H, W)

    # determine max depth for colorbar
    vmax = np.nanmax([depths_rgbd,depths_tof])
    vmin = 0

    # make single pixels visible
    # depths_tof = skimage.measure.block_reduce(depths_tof, (1,8,8), np.nanmax) # (N, H, W)
    # mask = skimage.measure.block_reduce(mask, (8,8), np.nanmax) # (H, W)
    # error_mask = skimage.measure.block_reduce(error_mask, (8,8), np.nanmax) # (H, W)
    # mask_comb = skimage.measure.block_reduce(mask_comb, (8,8), np.nanmax) # (H, W)

    mask = grey_dilation(mask, size=(args.tof.tof_pix_size,args.tof.tof_pix_size)) # (H, W)
    error_mask = grey_dilation(error_mask, size=(args.tof.tof_pix_size,args.tof.tof_pix_size)) # (H, W)
    mask_comb = grey_dilation(mask_comb, size=(args.tof.tof_pix_size,args.tof.tof_pix_size)) # (H, W)

    for i in range(num_imgs):
        ax = axes[0,i]
        im = ax.imshow(depths_rgbd[i], vmin=vmin, vmax=vmax, cmap='jet')
        ax.set_title(f'Depth Map GT: {i}')

        ax = axes[1,i]
        im = ax.imshow(depths_tof[i], vmin=vmin, vmax=vmax, cmap='jet')
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
    
    # add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout()
    plt.show()


def test_USSModel():
    num_imgs = 3
    
    # create dataset
    args = Args(
        file_name="ethz_usstof_win.json"
    )
    args.dataset.keep_N_observations = num_imgs
    args.training.sampling_strategy = {
        "imgs": "same",
        "rays": "entire_img",
    } 
    args.training.sensors = ["USS", "RGBD"]
    args.dataset.sensors = ["USS", "RGBD"]

    if args.dataset.name == "RH2":
        dataset_class = DatasetRH
    elif args.dataset.name == "ETHZ":
        dataset_class = DatasetETHZ
    dataset = dataset_class(
        args = args,
        split="train",
    )
    img_wh = dataset.img_wh

    # get depths
    depths_uss = np.zeros((num_imgs, img_wh[0]*img_wh[1]))
    depths_rgbd = np.zeros((num_imgs, img_wh[0]*img_wh[1]))
    for i, j in enumerate(np.linspace(0, len(dataset)-1, num_imgs, dtype=int)):
        data = dataset(
            batch_size=img_wh[0]*img_wh[1],
            sampling_strategy=args.training.sampling_strategy,
            origin="nerf",
        )
        depths_uss[i] = data['depth']['USS'].detach().cpu().numpy()
        depths_rgbd[i] = data['depth']['RGBD'].detach().cpu().numpy()
        
    mask = dataset.sensors_dict['USS1'].mask.cpu().numpy().astype(int)

    # plot
    fig, axes = plt.subplots(ncols=num_imgs, nrows=3, figsize=(12,8))
    depths_rgbd = depths_rgbd.reshape(num_imgs, img_wh[1], img_wh[0])
    depths_uss = depths_uss.reshape(num_imgs, img_wh[1], img_wh[0])
    mask = mask.reshape(img_wh[1], img_wh[0])
    vmax = np.nanmax([depths_rgbd,depths_uss])

    # # make single pixels visible
    # depths_out = skimage.measure.block_reduce(depths_out, (1,4,4), np.nanmax) # (N, h, w)

    for i in range(num_imgs):
        ax = axes[0,i]
        im = ax.imshow(depths_rgbd[i], vmin=0, vmax=vmax, cmap='jet')
        ax.set_title(f'Image {i}')
        if i == 0:
            ax.set_ylabel(f'RGBD depth')
        if i == num_imgs-1:
            fig.colorbar(im, ax=ax, location='right')

        ax = axes[1,i]
        im = ax.imshow(mask, vmin=0, vmax=1, cmap='jet')
        if i == 0:
            ax.set_ylabel(f'Mask')
        if i == num_imgs-1:
            fig.colorbar(im, ax=ax, location='right')

        ax = axes[2,i]
        im = ax.imshow(depths_uss[i], vmin=0, vmax=vmax, cmap='jet')
        if i == 0:
            ax.set_ylabel(f'USS depth')
        if i == num_imgs-1:
            fig.colorbar(im, ax=ax, location='right')
    
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