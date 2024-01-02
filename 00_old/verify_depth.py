import numpy as np
import torch
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from datasets.RH2 import RobotAtHomeDataset
from datasets.ray_utils import get_rays

def depth2pos(depths, rays_o, rays_d):
    """
    Convert camera depth measurements to 3D positions.
    Returns for invalid depth values (nan) again np.nan.
    Args:
        depths: (N,) depth measurements
        rays_o: (N, 3) camera positions
        rays_d: (N, 3) ray directions
    Returns:
        pos_nan: (N, 3) 3D positions in world coordinates
        val_idxs: valid indices / not np.nan of depths; bool array (N,)
    """
    # normalize directions
    rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)

    # get valid indices that are not nan
    val_idxs = np.isnan(depths)
    val_idxs = np.invert(val_idxs)

    # convert depth to 3D position
    pos_c = depths[val_idxs, None] * rays_d[val_idxs]
    pos_w = pos_c + rays_o[val_idxs]

    # incert nan where depth is not given
    pos_nan = np.full(rays_o.shape, np.nan)
    pos_nan[val_idxs] = pos_w

    return pos_nan, val_idxs

def plotMaps(slice_map, depth_maps, scales_max, biases_max):
    extent = [-0.5, 0.5, -0.5, 0.5]

    fig, axes = plt.subplots(ncols=3, nrows=len(depth_maps), figsize=(8,8))

    for i in range(len(depth_maps)):
        
        comb_map = slice_map + 2*depth_maps[i]
        score = np.sum(slice_map * depth_maps[i]) / np.sum(depth_maps[i])

        ax = axes[i,0]
        ax.imshow(slice_map, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=3)
        ax.set_title(f'Slice Map (scale={scales_max[i]:.4f})')

        ax = axes[i,1]
        ax.imshow(2*depth_maps[i], extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=3)
        ax.set_title(f'Depth Map (bias={biases_max[i]:.4f})')

        ax = axes[i,2]
        ax.imshow(comb_map, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=3)
        ax.set_title(f'Combined Map (score={score:.4f})')

    plt.tight_layout()
    plt.show()

def verifyDepth():
    
    # load dataset
    dataset = RobotAtHomeDataset(root_dir='../RobotAtHome2/data', sensor_name="RGBD_1")
    dataset.batch_size = dataset.img_wh[0] * dataset.img_wh[1]
    dataset.ray_sampling_strategy = 'same_image'
    dataset.pixel_sampling_strategy = 'entire_image'

    # get ground truth
    slice_height = 1.045
    height_tolerance = 0.1
    slice_res = 512
    slice_map = dataset.getSceneSlice(height=slice_height, slice_res=slice_res, height_tolerance=height_tolerance)



    # scales = 3.5 * np.array([1.31875, 1.321875, 1.325, 1.327875, 1.33075]) / 115.0
    # scales = 3.5 * np.array([0.98, 0.99, 1.0, 1.01, 1.02]) / 115.0

    scales = np.linspace(4.4, 4.8, 11)
    biases = np.linspace(0, 0.15, 4)

    # create depth map
    depth_maps = np.empty((len(scales), len(biases), slice_res, slice_res))
    scores = np.zeros((len(scales), len(biases)))
    
    depth_max = 0
    nb_imgs = dataset.rays.shape[0]
    with alive_bar(scales.shape[0], bar = 'bubbles', receipt=False) as bar:
        for s, scale in enumerate(scales):
            for b, bias in enumerate(biases):
                
                for i in range(nb_imgs):
                    # get ray origins and directions
                    data = dataset[i]
                    rays_o, rays_d = get_rays(data['direction'], data['pose'].reshape(3,4))

                    # convert to numpy and reshape
                    rays_o = rays_o.detach().clone().numpy() # (H*W, 3)
                    rays_d = rays_d.detach().clone().numpy() # (H*W, 3)
                    depths = data["depth"].detach().clone().numpy() # (H*W)

                    # extract horizontal rays
                    w, h = dataset.img_wh
                    rays_o = rays_o.reshape((h,w,3))[h//2,:,:] # (W, 3)
                    rays_d = rays_d.reshape((h,w,3))[h//2,:,:] # (W, 3)
                    depths = depths.reshape((h,w))[h//2,:] # (W)

                    # transform depth
                    if depth_max < np.nanmax(depths):
                        depth_max = np.nanmax(depths)
                    depths = scale * depths / 115.0 + bias
                    depths = dataset.scalePosition(depths, only_scale=True)

                    # find position of depth values
                    depth_pos, val_idxs = depth2pos(depths, rays_o, rays_d) # (H*W, 3)
                    depth_pos = depth_pos[val_idxs] # (N, 3)

                    # update depth map
                    depth_idxs = np.round((depth_pos+0.5)*slice_res).astype(np.int32) # (N, 3)
                    depth_idxs = np.clip(depth_idxs, 0, slice_res-1)
                    depth_maps[s,b,depth_idxs[:,0], depth_idxs[:,1]] = 1

                # compute score
                scores[s,b] = np.sum(slice_map * depth_maps[s,b]) / np.sum(depth_maps[s,b])
            bar()

    print(f"depth_max: {depth_max}")

    # plot scores
    fig, ax = plt.subplots()
    im1 = ax.imshow(scores, extent=[biases[0], biases[-1], scales[0], scales[-1]], origin='lower', cmap='viridis')
    ax.set_xlabel('Bias')
    ax.set_ylabel('Scale')
    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('Score')
    plt.show()

    # plot
    plot_depth_maps = []
    scales_max = []
    biases_max = []
    for b in range(len(biases)):
        idx = np.argmax(scores[:,b])
        plot_depth_maps.append(depth_maps[idx,b])
        scales_max.append(scales[idx])
        biases_max.append(biases[b])
    plotMaps(slice_map, plot_depth_maps, scales_max, biases_max)

if __name__ == '__main__':
    verifyDepth()