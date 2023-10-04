import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
import os
import sys
 
sys.path.insert(0, os.getcwd())
from datasets.sensor_model import ToFModel, USSModel, SimpleUSSModel
from datasets.robot_at_home import RobotAtHomeDataset
from args.args import Args


def test_ToFModel():
    num_imgs = 3

    # img_wh = (32,40)
    # depths = 100 * np.random.rand(num_imgs, img_wh[1]*img_wh[0])

    root_dir =  '../RobotAtHome2/data'   
    dataset = RobotAtHomeDataset(
        root_dir=root_dir,
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
        
    model = ToFModel(img_wh)
    depths_out = model.convertDepth(depths=depths)

    print(np.nanmax(depths))
    print(np.nanmin(depths))
    print(np.nanmax(depths_out))
    print(np.nanmin(depths_out))


    # plot
    fig, axes = plt.subplots(ncols=depths.shape[0], nrows=2, figsize=(12,8))
    depths = depths.reshape(num_imgs, img_wh[1], img_wh[0])
    depths_out = depths_out.reshape(num_imgs, img_wh[1], img_wh[0])

    # make single pixels visible
    depths_out = skimage.measure.block_reduce(depths_out, (1,8,8), np.nanmax) # (N, h, w)

    for i in range(depths.shape[0]):
        ax = axes[0,i]
        ax.imshow(depths[i])
        ax.set_title(f'Depth Map GT: {i}')

        ax = axes[1,i]
        ax.imshow(depths_out[i])
        ax.set_title(f'Depth Map ToF: {i}')
    
    plt.tight_layout()
    plt.show()


def test_USSModel():
    num_imgs = 3
    
    # img_wh = (8, 8)
    # depths = 100 * np.random.rand(num_imgs, img_wh[0]*img_wh[1])
       
    root_dir =  '../RobotAtHome2/data'   
    dataset = RobotAtHomeDataset(
        root_dir=root_dir,
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
        
    model = USSModel(img_wh)
    depths_out, depths_prob = model.convertDepth(depths, return_prob=True)

    # plot
    fig, axes = plt.subplots(ncols=4, nrows=num_imgs, figsize=(12,8))
    depths = depths.reshape(num_imgs, img_wh[1], img_wh[0])
    depths_out = depths_out.reshape(num_imgs, model.h, model.w)
    depths_prob = depths_prob.reshape(num_imgs, model.h, model.w)
    gaussian = model.gaussian.reshape(model.h, model.w)

    for i in range(depths.shape[0]):
        ax = axes[i,0]
        ax.imshow(depths[i], vmin=0, vmax=np.nanmax(depths))
        ax.set_title(f'Depth Map GT')

        ax = axes[i,1]
        ax.imshow(gaussian, vmin=0, vmax=1)
        ax.set_title(f'Gaussian')

        ax = axes[i,2]
        ax.imshow(depths_prob[i], vmin=0, vmax=depths_prob[i].max())
        ax.set_title(f'Depth Probability')

        ax = axes[i,3]
        ax.imshow(depths_out[i], vmin=0, vmax=np.nanmax(depths))
        ax.set_title(f'Depth down sampled')
    
    plt.tight_layout()
    plt.show()

def test_SimpleUSSModel():
    num_imgs = 3
    
    # img_wh = (64, 64)
    # depths = 100 * np.random.rand(num_imgs, img_wh[0]*img_wh[1])
    
    args = Args(file_name="hparams.json")
    args.rh.sensor_model = False

    root_dir =  '../RobotAtHome2/data'   
    dataset = RobotAtHomeDataset(
        args = args,
        root_dir=root_dir,
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
        
    model = SimpleUSSModel(img_wh)
    depths_out = model.convertDepth(depths)

    # plot
    fig, axes = plt.subplots(ncols=3, nrows=num_imgs, figsize=(12,8))
    depths = depths.reshape(num_imgs, img_wh[1], img_wh[0])
    depths_out = depths_out.reshape(num_imgs, img_wh[1], img_wh[0])
    mask = model.mask.reshape(img_wh[1], img_wh[0])

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
    # test_USSModel()
    # test_ToFModel()
    test_SimpleUSSModel()