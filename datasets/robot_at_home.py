import glob
import os

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from robotathome import RobotAtHome
from robotathome import logger, log
from robotathome import time_win2unixepoch, time_unixepoch2win



try:
    from .ray_utils import get_rays
    from .base import BaseDataset
    from .color_utils import read_image
    from .ray_utils import get_ray_directions
except:
    from ray_utils import get_rays
    from base import BaseDataset
    from color_utils import read_image
    from ray_utils import get_ray_directions


class RobotAtHomeDataset(BaseDataset):

    def __init__(self, root_dir, split='train', downsample=1.0, sensor_name="all", **kwargs):
        super().__init__(root_dir, split, downsample)

        # self.read_intrinsics()
        # if kwargs.get('read_meta', True):
        #     xyz_min, xyz_max = \
        #         np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
        #     self.shift = (xyz_max + xyz_min) / 2
        #     self.scale = (xyz_max -
        #                   xyz_min).max() / 2 * 1.05  # enlarge a little

        #     # hard-code fix the bound error for some scenes...
        #     if 'Mic' in self.root_dir:
        #         self.scale *= 1.2
        #     elif 'Lego' in self.root_dir:
        #         self.scale *= 1.1
        #     self.read_meta(split)

        my_rh_path = root_dir
        my_rgbd_path = os.path.join(my_rh_path, 'files/rgbd')
        my_scene_path = os.path.join(my_rh_path, 'files/scene')
        my_wspc_path = 'results'
        my_db_filename = "rh.db"
        self.rh = RobotAtHome(rh_path=my_rh_path, rgbd_path=my_rgbd_path, scene_path=my_scene_path, wspc_path=my_wspc_path, db_filename=my_db_filename)

        # get only observations from home="alma" and room="livingroom"
        df = self.rh.get_sensor_observations('lblrgbd') # load only labeld RGBD observations
        home_id = self.rh.name2id("anto", "h")
        room_id = self.rh.name2id("anto_bedroom1", "r")
        self.df = df[(df['home_id'] == home_id) & (df['room_id'] == room_id)]

        if sensor_name != "all":
            sensor_id = self.rh.name2id(sensor_name, "s")
            self.df = self.df[self.df["sensor_id"] == sensor_id]

        # split dataset
        self.splitDataset()

        self.img_wh, self.K, self.directions = self.read_intrinsics()
        self.rays, self.poses = self.read_meta(split)

    def read_intrinsics(self):
        """
        Read camera intrinsics from the dataset.
        Returns:
            img_wh: tuple of image width and height
            K: camera intrinsic matrix; tensor of shape (3, 3)
            directions: tensor of shape (W*H, 3) containing ray directions
        """
        # get image hight and width
        id = self.df["id"].to_numpy()[0]
        [rgb_f, d_f] = self.rh.get_RGBD_files(id)
        img = mpimg.imread(rgb_f)
        h, w, _ = img.shape

        # get camera intrinsic matrix
        # f = 570.3422241210938 # focal length
        fh = 232.6282 # 232.632 if h/2
        fw = 220.7983 # 220.8053 if w/2
        K = np.array([[fw, 0.0, w/2-0.5],
                      [0.0, fh, h/2-0.5], 
                      [0.0, 0.0, 1.0]])

        # get ray directions
        directions = get_ray_directions(h, w, K) # (H*W, 3) = (H, W, 3).flatten()

        # # verify opening angle
        # W_mid = w // 2
        # H_mid = h // 2
        # directions_np = directions.detach().cpu().numpy().reshape(h, w, 3)
        # alpha_H = np.arccos(np.dot(directions_np[0,W_mid,:], directions_np[-1,W_mid,:]))
        # alpha_W = np.arccos(np.dot(directions_np[H_mid,0,:], directions_np[H_mid,-1,:]))
        # print(f"opening angle height = {np.rad2deg(alpha_H)}")
        # print(f"opening angle width = {np.rad2deg(alpha_W)}")

        return (w, h), torch.FloatTensor(K), directions

    def read_meta(self, split):
        """
        Read meta data from the dataset.
        Args:
            split: string indicating which split to read from
        Returns:
            rays: tensor of shape (N_images, W*H, 3) containing RGB images
            poses: tensor of shape (N_images, 3, 4) containing camera poses
        """
        df = self.df[self.df["split"] == split].copy(deep=True)

        # get images
        rays = np.empty(())
        ids = df["id"].to_numpy()
        rays = np.empty((ids.shape[0], self.img_wh[0]*self.img_wh[1], 3))
        for i, id in enumerate(ids):
            [rgb_f, d_f] = self.rh.get_RGBD_files(id)
            rays[i,:,:] = mpimg.imread(rgb_f).reshape(self.img_wh[0]*self.img_wh[1], 3)

        # get position
        sensor_pose_x = df["sensor_pose_x"].to_numpy()
        sensor_pose_y = df["sensor_pose_y"].to_numpy()
        sensor_pose_z = df["sensor_pose_z"].to_numpy()
        p_c2w = np.stack((sensor_pose_x, sensor_pose_y, sensor_pose_z), axis=1)

        # get orientation
        sensor_pose_yaw = df["sensor_pose_yaw"].to_numpy()
        sensor_pose_pitch = df["sensor_pose_pitch"].to_numpy()
        sensor_pose_roll = df["sensor_pose_roll"].to_numpy()

        # sensor_pose_pitch += np.deg2rad(90)
        # sensor_pose_roll += np.deg2rad(90)
        sensor_pose_yaw -= np.deg2rad(90)
        # sensor_pose_yaw = 0.0 * np.ones_like(sensor_pose_yaw)
        # sensor_pose_pitch = 0.0 * np.ones_like(sensor_pose_pitch)
        # sensor_pose_roll = - np.deg2rad(90) * np.ones_like(sensor_pose_roll)

        R_yaw_c2w = np.stack((np.cos(sensor_pose_yaw), -np.sin(sensor_pose_yaw), np.zeros_like(sensor_pose_yaw),
                              np.sin(sensor_pose_yaw), np.cos(sensor_pose_yaw), np.zeros_like(sensor_pose_yaw),
                              np.zeros_like(sensor_pose_yaw), np.zeros_like(sensor_pose_yaw), np.ones_like(sensor_pose_yaw)), axis=1).reshape(-1, 3, 3)
        R_pitch_c2w = np.stack((np.cos(sensor_pose_pitch), np.zeros_like(sensor_pose_pitch), np.sin(sensor_pose_pitch),
                                np.zeros_like(sensor_pose_pitch), np.ones_like(sensor_pose_pitch), np.zeros_like(sensor_pose_pitch),
                                -np.sin(sensor_pose_pitch), np.zeros_like(sensor_pose_pitch), np.cos(sensor_pose_pitch)), axis=1).reshape(-1, 3, 3)
        R_roll_c2w = np.stack((np.ones_like(sensor_pose_roll), np.zeros_like(sensor_pose_roll), np.zeros_like(sensor_pose_roll),
                               np.zeros_like(sensor_pose_roll), np.cos(sensor_pose_roll), -np.sin(sensor_pose_roll),
                               np.zeros_like(sensor_pose_roll), np.sin(sensor_pose_roll), np.cos(sensor_pose_roll)), axis=1).reshape(-1, 3, 3)
        R_c2w = np.matmul(R_yaw_c2w, np.matmul(R_pitch_c2w, R_roll_c2w))
        # R_c2w = np.matmul(R_roll_c2w, np.matmul(R_pitch_c2w, R_yaw_c2w))

        poses = np.concatenate((R_c2w, p_c2w[:, :, np.newaxis]), axis=2)

        # # translate position TODO
        # poses[:,:,3] -= self.shift
        # poses[:,:,3] /= 2 * self.scale  # to bound the scene inside [-0.5, 0.5]

        return torch.tensor(rays, dtype=torch.float32), torch.tensor(poses, dtype=torch.float32)

    def splitDataset(self):
        """
        Split the dataset into train, val and test sets.
        """
        df = self.df.copy(deep=True)

        # create new column for split
        df.insert(1, 'split', None)

        # get indices for each sensor_id and each split
        train_idxs = np.empty(0, dtype=int)
        val_idxs = np.empty(0, dtype=int)
        test_idxs = np.empty(0, dtype=int)
        for id in df["sensor_id"].unique():
            id_idxs = df.index[df["sensor_id"] == id].to_numpy()

            train_idxs = np.concatenate((train_idxs, id_idxs[::3]))
            val_idxs = np.concatenate((val_idxs, id_idxs[1::3]))
            test_idxs = np.concatenate((test_idxs, id_idxs[2::3]))

        # set split column    
        df.loc[train_idxs, 'split'] = 'train'
        df.loc[val_idxs, 'split'] = 'val'
        df.loc[test_idxs, 'split'] = 'test'

        self.df = df




def createFoVpolygon(corners, rays):
    # scale rays
    rays = 0.3 * ( rays.T / np.linalg.norm(rays, axis=1) ).T

    # Calculate field of view vertices
    fov_vertices1 = np.zeros((5, 3))
    fov_vertices1[0, :] = corners[0, :]  # Start from the first corner
    for j in range(1, 4):
        fov_vertices1[j, :] = corners[j, :]
    fov_vertices1[4, :] = corners[0, :]  # Close the polygon by going back to the first corner

    fov_vertices2 = np.zeros((5, 3))
    fov_vertices2[0, :] = corners[0, :]
    fov_vertices2[1, :] = corners[0, :] + rays[0, :] 
    fov_vertices2[2, :] = corners[1, :] + rays[1, :]
    fov_vertices2[3, :] = corners[1, :]
    fov_vertices2[4, :] = corners[0, :]

    fov_vertices3 = np.zeros((5, 3))
    fov_vertices3[0, :] = corners[1, :]
    fov_vertices3[1, :] = corners[1, :] + rays[1, :]
    fov_vertices3[2, :] = corners[2, :] + rays[2, :]
    fov_vertices3[3, :] = corners[2, :]
    fov_vertices3[4, :] = corners[1, :]

    fov_vertices4 = np.zeros((5, 3))
    fov_vertices4[0, :] = corners[2, :]
    fov_vertices4[1, :] = corners[2, :] + rays[2, :]
    fov_vertices4[2, :] = corners[3, :] + rays[3, :]
    fov_vertices4[3, :] = corners[3, :]
    fov_vertices4[4, :] = corners[2, :]

    fov_vertices5 = np.zeros((5, 3))
    fov_vertices5[0, :] = corners[3, :]
    fov_vertices5[1, :] = corners[3, :] + rays[3, :]
    fov_vertices5[2, :] = corners[0, :] + rays[0, :]
    fov_vertices5[3, :] = corners[0, :]
    fov_vertices5[4, :] = corners[3, :]

    fov_vertices6 = np.zeros((5, 3))
    fov_vertices6[0, :] = corners[0, :] + rays[0, :] 
    for j in range(1, 4):
        fov_vertices6[j, :] = corners[j, :] + rays[j, :] 
    fov_vertices6[4, :] = corners[0, :] + rays[0, :] 

    # Create a polygon to represent the field of view (transparent)
    fov_polygon = [fov_vertices1, fov_vertices2, fov_vertices3, fov_vertices4, fov_vertices5, fov_vertices6]

    return fov_polygon

def plotCameraFoV(rays_o, rays_d, color, ax):
    """
    Visualize camera viewing directions.
    """

    # plot field of view of camera
    all_corners = np.array([rays_o[0,0,:], rays_o[-1,0,:], 
                            rays_o[-1,-1,:], rays_o[0,-1,:]]) # (4, 3)
    all_rays = np.array([rays_d[0,0,:], rays_d[-1,0,:], 
                            rays_d[-1,-1,:], rays_d[0,-1,:]]) # (4, 3)
    fov_polygon = createFoVpolygon(all_corners, all_rays)
    ax.add_collection3d(Poly3DCollection(fov_polygon, alpha=0.18, edgecolor=color, facecolor=color))

    # plot top left corner of field of view
    top_left_corner = rays_o[0,0,:] + 0.3*rays_d[0,0,:] / np.linalg.norm(rays_d[0,0,:])
    ax.scatter(top_left_corner[0], top_left_corner[1], top_left_corner[2], color=color)

    bottom_left_corner = rays_o[-1,0,:] + 0.3*rays_d[-1,0,:] / np.linalg.norm(rays_d[-1,0,:])
    ax.scatter(bottom_left_corner[0], bottom_left_corner[1], bottom_left_corner[2], color=color, marker='x')


def plotCameraRays(rays_o, rays_d, ax, color, show_nb_rays):
    """
    Visualize sampled rays.
    """
    idx_w = np.random.randint(0, rays_o.shape[0]-1, show_nb_rays)
    idx_h = np.random.randint(0, rays_o.shape[1]-1, show_nb_rays)

    # normalize ray directions
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)

    # Plot camera positions as dots with color gradient
    for i in range(show_nb_rays):

        # plot all sample rays
        start = rays_o[idx_w[i], idx_h[i], :]
        end = start + 0.5 * rays_d[idx_w[i], idx_h[i], :]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c=color, label='Ray Direction')

def plotCameraDirections(df, ax, color):
    # get pose
    x = df["sensor_pose_x"]
    y = df["sensor_pose_y"]
    z = df["sensor_pose_z"]
    yaw = df["sensor_pose_yaw"]

    R_yaw_c2w = np.array([[np.cos(yaw), -np.sin(yaw), np.zeros_like(yaw)],
                            [np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)],
                            [np.zeros_like(yaw), np.zeros_like(yaw), np.ones_like(yaw)]])
    
    # Plot points
    ax.scatter(x, y, z, label='Points', color=color)
    
    # Plot directions
    direction = np.matmul(R_yaw_c2w[:,:], np.array([1, 0, 0]))
    ax.quiver(x, y, z, direction[0], direction[1], direction[2], length=0.1, normalize=True, color=color)


def test_read_meta():
    log.set_log_level('INFO')  # SUCCESS is the default

    show_nb_cameras = 7
    show_nb_rays = 10

    # create figure
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    
    sensor_names = ("RGBD_1", "RGBD_2", "RGBD_3", "RGBD_4")
    color_map_names = ("Purples", "Greens", "Reds", "Greys")
    for j, (sensor_name, color_map_name) in enumerate(zip(sensor_names, color_map_names)):
        dataset = RobotAtHomeDataset(root_dir='../RobotAtHome2/data', sensor_name=sensor_name)
        dataset.batch_size = show_nb_rays
        dataset.ray_sampling_strategy = 'same_image'

        df_split_idxs = dataset.df[dataset.df["split"] == "train"].index.to_numpy()
        df_split = dataset.df.loc[df_split_idxs].copy(deep=True)

        # create color bar
        cmap = plt.get_cmap(color_map_name)
        norm = Normalize(vmin=0, vmax=1.5*dataset.rays.shape[0])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        colour_map = {"cmap": cmap, "norm": norm, "sm": sm}
        sm.set_array([])
        # if j == 0:
        #     cbar = plt.colorbar(sm, ax=ax, label='Pose Index')
        # else:
        #     cbar = plt.colorbar(sm, ax=ax)

        for i in np.linspace(0, dataset.rays.shape[0]-1, show_nb_cameras, dtype=int):
            # get ray origins and directions
            data = dataset[i]
            rays_o, rays_d = get_rays(data['direction'], data['pose'].reshape(3,4)) # (W*H, 3), (3,4)
            rays_o = rays_o.reshape(dataset.img_wh[1], dataset.img_wh[0], 3).detach().clone().numpy() # (H, W, 3)
            rays_d = rays_d.reshape(dataset.img_wh[1], dataset.img_wh[0], 3).detach().clone().numpy() # (H, W, 3)

            # verify ratio height to width
            dH = np.linalg.norm(rays_d[0,0,:] - rays_d[-1,0,:])
            dW = np.linalg.norm(rays_d[0,0,:] - rays_d[0,-1,:])
            if np.abs(dH/dW - dataset.img_wh[1]/dataset.img_wh[0]) > 0.01:
                print(f"ERROR: test_read_meta: Ratio height to width is not correct! {dH/dW} != {dataset.img_wh[1]/dataset.img_wh[0]}")

            # # veryfy rays_d
            # directions = get_ray_directions(dataset.img_wh[1], dataset.img_wh[0], dataset.K, flatten=False).detach().clone().numpy() # (H, W, 3)
            # yaw = dataset.df["sensor_pose_yaw"].to_numpy()[i]
            # R_yaw_c2w = np.array([[np.cos(yaw), -np.sin(yaw), np.zeros_like(yaw)],
            #                         [np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)],
            #                         [np.zeros_like(yaw), np.zeros_like(yaw), np.ones_like(yaw)]])
            # directions = np.einsum('hwi,ji->hwj', directions, R_yaw_c2w)
            # if not np.allclose(rays_d, directions, atol=1e-6):
            #     print(f"ERROR: test_read_meta: Rays_d is not correct!, error = {np.linalg.norm(rays_d-directions)}")
            #     print(f"rays_d[0,0,:] = {rays_d[0,0,:]}")
            #     print(f"directions[0,0,:] = {directions[0,0,:]}")

            # verify opening angle
            W_mid = dataset.img_wh[0] // 2
            H_mid = dataset.img_wh[1] // 2
            alpha_H = np.arccos(np.dot(rays_d[0,W_mid,:], rays_d[-1,W_mid,:]))
            alpha_W = np.arccos(np.dot(rays_d[H_mid,0,:], rays_d[H_mid,-1,:]))
            print(f"opening angle height = {np.rad2deg(alpha_H)}")
            print(f"opening angle width = {np.rad2deg(alpha_W)}")

            color = colour_map["cmap"](colour_map["norm"](i+int(dataset.rays.shape[0]/2)))

            plotCameraFoV(rays_o, rays_d, color, ax)
            # plotCameraRays(rays_o, rays_d, ax, color, show_nb_rays)      
            plotCameraDirections(df_split.iloc[i], ax, color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses Visualization')
    # ax.set_aspect('equal', 'box')
    ax.set_box_aspect((3, 3, 1.5))
    ax.set_xlim(0, 3)
    ax.set_xticks(np.linspace(0, 3, 7))
    ax.set_ylim(-2, 1)
    ax.set_yticks(np.linspace(-2, 1, 7))
    ax.set_zlim(0, 1.5)
    ax.set_zticks(np.linspace(0, 1.5, 4))
    plt.show()



if __name__ == '__main__':
    test_read_meta()