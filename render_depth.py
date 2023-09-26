import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from alive_progress import alive_bar

from datasets.robot_at_home import RobotAtHomeDataset
from datasets.ray_utils import get_rays


def convertDepth2Pos(depths, rays_o, rays_d):
    """
    Convert camera depth measurements to 3D positions
    Args:
        depths: (N,) depth measurements
        rays_o: (N, 3) camera positions
        rays_d: (N, 3) ray directions
    Returns:
        pos_w: (N, 3) 3D positions in world coordinates
    """
    # normalize directions
    rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)

    # convert depth to 3D position
    pos_c = depths[:, None] * rays_d
    pos_w = pos_c + rays_o

    return pos_w

def findClosestPoints(array1, array2):
    """
    Find the closest points in array2 for each point in array1
    and return the indices of array2 for each point in array1.
    Args:
        array1: (N, 3) array of points
        array2: (M, 3) array of points
    Returns:
        closest_indices: (N,) indices of array2 from closest points
    """
    # downsample arrays
    array1 = np.copy(array1.astype(np.float32))
    array2 = np.copy(array2.astype(np.float32))

    # define batch size
    batch_size = 100
    while array1.shape[0]%batch_size != 0:
        batch_size -= 1

    # split calculation in pieces to avoid memory error
    closest_indices = np.zeros(array1.shape[0], dtype=np.int32) # (N,)
    with alive_bar(array1.shape[0]//batch_size, bar = 'bubbles', receipt=False) as bar:
        for i in range(0, array1.shape[0], batch_size):
            distances = np.linalg.norm(array2[:, np.newaxis] - array1[i:i+batch_size], axis=2) # (M, batch_size)
            closest_indices[i:i+batch_size] = np.argmin(distances, axis=0)
            bar()

    # distances = np.linalg.norm(array2[:, np.newaxis] - array1, axis=2) # (M, N)
    # closest_indices = np.argmin(distances, axis=0)
    
    return closest_indices

def renderScene(dataset, depths, rays_o, rays_d):
    """
    Render scene color given depth measurements and camera positions and viewing directions.
    Args:
        dataset: RobotAtHomeDataset
        depths: (N,) depth measurements
        rays_o: (N, 3) camera positions
        rays_d: (N, 3) ray directions
    Returns:
        rgbs: (N, 3) scene color
    """
    # convert depth to 3D position
    pos = convertDepth2Pos(depths=depths, rays_o=rays_o, rays_d=rays_d)

    # get scene point cloud
    scene_file = dataset.scene.scene_file.values[0]
    scene_point_cloud = np.loadtxt(scene_file, skiprows=6)

    # limit scene point cloud to 3D positions within the camera frustum
    print(f"start: {scene_point_cloud.shape}")
    pos_min = pos.min(axis=0) - 0.05
    pos_max = pos.max(axis=0) + 0.05
    scene_point_cloud = scene_point_cloud[(scene_point_cloud[:,0] > pos_min[0]) & (scene_point_cloud[:,0] < pos_max[0]) \
                                        & (scene_point_cloud[:,1] > pos_min[1]) & (scene_point_cloud[:,1] < pos_max[1]) \
                                        & (scene_point_cloud[:,2] > pos_min[2]) & (scene_point_cloud[:,2] < pos_max[2])]
    print(f"end: {scene_point_cloud.shape}")

    # render color of closest points
    closest_idxs = findClosestPoints(array1=pos, array2=scene_point_cloud[:,:3])
    rgbs = scene_point_cloud[closest_idxs, 3:]

    return rgbs

def plotImg(rgbs_gt, rgbs_scene, depths):
    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    # Plot the first image on the first axis
    axs[0].imshow(rgbs_gt)  
    axs[0].set_title('Ground Truth')

    # Plot the second image on the second axis
    axs[1].imshow(rgbs_scene)  
    axs[1].set_title('Scene render using depth')

    # Plot the second image on the second axis 
    d = axs[2].imshow(depths, cmap='viridis')
    fig.colorbar(d, ax=axs[2])
    axs[2].set_title('Scene render using depth')

    # Remove axis ticks and labels for images
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')

    # Adjust layout to prevent axis labels from overlapping
    plt.tight_layout()

    # Display the plot
    plt.show()

def renderDepth():
    # load dataset
    dataset = RobotAtHomeDataset(root_dir='../RobotAtHome2/data', sensor_name="RGBD_1")
    dataset.batch_size = dataset.img_wh[0] * dataset.img_wh[1]
    dataset.ray_sampling_strategy = 'same_image'
    dataset.pixel_sampling_strategy = 'entire_image'

    for i in range(dataset.rays.shape[0]):
        # get ray origins and directions
        data = dataset[i]
        rays_o, rays_d = get_rays(data['direction'], data['pose'].reshape(3,4))

        # convert to numpy and reshape
        rays_o = rays_o.detach().clone().numpy() # (H*W, 3)
        rays_d = rays_d.detach().clone().numpy() # (H*W, 3)
        depths = data["depth"].detach().clone().numpy() # (H*W)
        rgbs_gt = data["rgb"].detach().clone().numpy() # (H*W, 3)

        # downsample arrays
        dw = 1 # downsample factor
        W, H = dataset.img_wh
        w, h = W//dw, H//dw
        rays_o = rays_o.reshape(H, W, 3)[::dw, ::dw].reshape(-1, 3) # (H*W, 3)
        rays_d = rays_d.reshape(H, W, 3)[::dw, ::dw].reshape(-1, 3)
        depths = depths.reshape(H, W)[::dw, ::dw].reshape(-1)
        rgbs_gt = rgbs_gt.reshape(H, W, 3)[::dw, ::dw].reshape(-1, 3)

        # render scene colors
        # rgbs_scene = renderScene(dataset=dataset, depths=depths, rays_o=rays_o, rays_d=rays_d)
        rgbs_scene = rgbs_gt

        # plot images
        plotImg(rgbs_gt=rgbs_gt.reshape(h, w, 3), 
                rgbs_scene=rgbs_scene.reshape(h, w, 3),
                depths=depths.reshape(h, w))




if __name__ == '__main__':
    renderDepth()