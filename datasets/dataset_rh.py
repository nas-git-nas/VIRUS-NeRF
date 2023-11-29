import glob
import os

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import cv2 as cv

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from robotathome import RobotAtHome
from robotathome import logger, log
from robotathome import time_win2unixepoch, time_unixepoch2win

from datasets.sensor_model import RGBDModel, ToFModel, USSModel

from args.args import Args
from training.sampler import Sampler

# try:
#     from .ray_utils import get_rays
#     from .base import DatasetBase
#     from .color_utils import read_image
#     from .ray_utils import get_ray_directions
#     from .robot_at_home_scene import RobotAtHomeScene
# except:

from datasets.ray_utils import get_rays, get_ray_directions
from datasets.dataset_base import DatasetBase
from datasets.color_utils import read_image
from datasets.scene_rh import SceneRH


class DatasetRH(DatasetBase):

    def __init__(
        self, 
        args:Args, 
        split:str='train',
        scene:SceneRH=None,
    ):

        super().__init__(args=args, split=split)

        # self.args = args

        # load dataset
        self.rh = RobotAtHome(
            rh_path = self.args.dataset.path, 
            rgbd_path = os.path.join(self.args.dataset.path, 'files/rgbd'), 
            scene_path = os.path.join(self.args.dataset.path, 'files/scene'), 
            wspc_path = 'results', 
            db_filename = "rh.db"
        )

        # load dataframe
        self.df = self._loadRHDataframe(split=split)

        # load scene
        if scene is None:
            self.scene = SceneRH(
                rh=self.rh, 
                args=self.args
            )
        else:
            self.scene = scene

        img_wh, K, directions = self.read_intrinsics()
        rays, depths, poses = self.read_meta(
            split=split,
            img_wh=img_wh,
        )

        if self.args.dataset.keep_pixels_in_angle_range != "all":
            rays, directions, depths, img_wh = self.reduceImgHeight(
                rays=rays,
                directions=directions,
                depths=depths,
                img_wh=img_wh,
                angle_min_max=self.args.dataset.keep_pixels_in_angle_range,
            )

        sensors_dict, depths_dict = self.createSensorModels(
            depths=depths,
            img_wh=img_wh,
        )

        self.img_wh = img_wh
        self.K = K
        self.poses = poses
        self.rays = rays
        self.directions = directions
        self.sensors_dict = sensors_dict
        self.depths_dict = depths_dict

        # TODO: move to base class
        self.sampler = Sampler(
            args=args,
            dataset_len=len(self),
            img_wh=self.img_wh,
            seed=args.seed,
            sensors_dict=self.sensors_dict,
            fct_getValidDepthMask=self.getValidDepthMask,
        )

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
        # # f = 570.3422241210938 / 2 # focal length
        # f = 363.83755595658255
        # fh = #232.6282 # 232.632 if h/2
        # fw = #220.7983 # 220.8053 if w/2
        # K = np.array([[f, 0.0, w/2],
        #               [0.0, f, h/2], 
        #               [0.0, 0.0, 1.0]])
        
        cx = 157.3245865
        cy = 120.0802295
        fx = 286.441384
        fy = 271.36999
        K = np.array([[fy, 0.0, cy],
                      [0.0, fx, cx], 
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

        return (w, h), torch.FloatTensor(K), directions.requires_grad_(requires_grad=False)

    def read_meta(
        self, 
        split:str,
        img_wh:tuple,
    ):
        """
        Read meta data from the dataset.
        Args:
            split: string indicating which split to read from; str
            img_wh: image width and height; tuple of ints
        Returns:
            rays: tensor of shape (N_images, H*W, 3) containing RGB images
            depths: tensor of shape (N_images, H*W) containing depth images
            poses: tensor of shape (N_images, 3, 4) containing camera poses
        """
        df = self.df[self.df["split"] == split].copy(deep=True)
        W, H = img_wh

        # get images
        rays = np.empty(())
        ids = df["id"].to_numpy()
        rays = np.empty((ids.shape[0], W*H, 3))
        depths = np.empty((ids.shape[0], W*H), dtype=np.float32)
        for i, id in enumerate(ids):
            [rgb_f, d_f] = self.rh.get_RGBD_files(id)
            rays[i,:,:] = mpimg.imread(rgb_f).reshape(W*H, 3)

            # depth_or = mpimg.imread(d_f)
            depth_or = cv.imread(d_f, cv.IMREAD_UNCHANGED)
            if np.max(depth_or) > 115 or np.min(depth_or) < 0:
                self.args.logger.error(f"robot_at_home.py: read_meta: depth image has invalid values")
            # depth = 4.7 * depth_or / 115.0 # convert to meters
            depth = 5.0 * depth_or / 128.0 # convert to meters

            if np.allclose(depth_or[:,:,0], depth_or[:,:,1]) and np.allclose(depth_or[:,:,0], depth_or[:,:,2]):
                depth = depth[:,:,0]
            else:
                self.args.logger.error(f"robot_at_home.py: read_meta: depth image has more than one channel")
            # depth = self.scalePosition(pos=depth, only_scale=True) # (H, W), convert to cube coordinate system [-0.5, 0.5]
            depth[depth==0] = np.nan # set invalid depth values to nan
            depth = self.scene.w2c(depth.flatten(), only_scale=True) # (H*W,), convert to cube coordinate system [-0.5, 0.5]
            depths[i,:] = depth

            # if i == 0 or i==50 or i==100 or i==200 or i==300 or i==400:
            #     fig, axs = plt.subplots(1, 3, figsize=(12, 6))

            #     axs[0].imshow(rays[i,:,:].reshape(self.img_wh[1], self.img_wh[0], 3))
            #     axs[0].set_title('Original Image')
            #     axs[0].axis('off')

            #     # Display the first image with the specified colormap and colorbar
            #     im1 = axs[1].imshow(depth_or[:,:,0], cmap='jet_r', aspect='equal')
            #     axs[1].set_title('Original Density')
            #     axs[1].axis('off')
            #     cbar1 = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            #     cbar1.set_label('Depth')

            #     # Display the second image with the specified colormap and colorbar
            #     im2 = axs[2].imshow(depth, cmap='jet_r', aspect='equal', vmin=0.0, vmax=4.7)
            #     axs[2].set_title('Transformed Density')
            #     axs[2].axis('off')
            #     cbar2 = plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
            #     cbar2.set_label('Depth')

            #     plt.tight_layout()
            #     plt.show()

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

        poses = np.concatenate((R_c2w, p_c2w[:, :, np.newaxis]), axis=2) # (N_images, 3, 4)

        # translate and scale position
        # poses[:,:,3] = self.scalePosition(pos=poses[:,:,3])
        poses[:,:,3] = self.scene.w2c(pos=poses[:,:,3], copy=False)

        return torch.tensor(rays, dtype=torch.float32, requires_grad=False), depths, torch.tensor(poses, dtype=torch.float32, requires_grad=False)
    
    def createSensorModels(
            self, 
            depths:torch.tensor,
            img_wh:tuple,
    ):
        """
        Create sensor models for each sensor and convert depths respectively.
        Args:
            depths: depths of all images; tensor of shape (N_images, H*W)
            img_wh: image width and height; tuple of ints
        Returns:
            sensors_dict: dictionary containing sensor models
            depths_dict: dictionary containing converted depths
        """
        sensors_dict = {} 
        for sensor_name in self.args.training.sensors:
            if sensor_name == "RGBD":
                sensors_dict["RGBD"] = RGBDModel(
                    args=self.args, 
                    img_wh=img_wh
                )
            elif sensor_name == "ToF":
                sensors_dict["ToF"] = ToFModel(
                    args=self.args, 
                    img_wh=img_wh
                )
            elif sensor_name == "USS":
                sensors_dict["USS"] = USSModel(
                    args=self.args, 
                    img_wh=img_wh,
                    num_imgs=self.df.shape[0],
                )
            else:
                print(f"ERROR: robot_at_home.__init__: sensor model {sensor_name} not implemented")

        depths_dict = {}
        for sensor_name, sensor_model in sensors_dict.items():
            depths_dict[sensor_name] = torch.tensor(
                data=sensor_model.convertDepth(depths),
                dtype=torch.float32,
                requires_grad=False,
            )

        return sensors_dict, depths_dict
    
    def getIdxFromSensorName(self, df, sensor_name):
        """
        Get the indices of the dataset that belong to a particular sensor.
        Args:
            df: robot@home dataframe, pandas df
            sensor_name: name of the sensor, str
        Returns:
            idxs: indices of the dataset that belong to the sensor
        """
        sensor_id = self.rh.name2id(sensor_name, "s")
        mask = np.array(df["sensor_id"] == sensor_id, dtype=bool)
        idxs = np.where(mask)[0]
        return idxs
    
    def _loadRHDataframe(self, split):
        """
        Load robot@home data frame
        Args:
            split: train, val or test split, str
        Returns:
            df: rh dataframe; pandas df
        """
        # load only labeld RGBD observations
        df = self.rh.get_sensor_observations('lblrgbd') 

        # get only observations from specific home and room 
        home_id = self.rh.name2id(self.args.rh.home, "h")
        room_id = self.rh.name2id(self.args.rh.home+"_"+self.args.rh.room, "r")
        df = df[(df['home_id'] == home_id) & (df['room_id'] == room_id)]

        # split dataset
        df = self.splitDataset(
            df = df, 
            split_ratio = self.args.dataset.split_ratio, 
            split_description_path = os.path.join(self.args.dataset.path, 'files', 'rgbd', 
                                                  self.args.rh.session, self.args.rh.home, self.args.rh.room),
            split_description_name = 'split_'+self.args.rh.subsession+'.csv'
        )
        df = df[df["split"] == split]

        # keep only observations from particular sensor
        if self.args.dataset.keep_sensor != "all":
            name_idxs = self.getIdxFromSensorName(df=df, sensor_name=self.args.dataset.keep_sensor)
            df = df[name_idxs]

        # keep only first N observations
        if self.args.dataset.keep_N_observations != "all":
            df = df.iloc[:self.args.dataset.keep_N_observations,:]

        return df
    

    



def createFoVpolygon(corners, rays):
    # scale rays
    rays = 0.05 * ( rays.T / np.linalg.norm(rays, axis=1) ).T

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
    top_left_corner = rays_o[0,0,:] + 0.05*rays_d[0,0,:] / np.linalg.norm(rays_d[0,0,:])
    ax.scatter(top_left_corner[0], top_left_corner[1], top_left_corner[2], color=color)

    bottom_left_corner = rays_o[-1,0,:] + 0.05*rays_d[-1,0,:] / np.linalg.norm(rays_d[-1,0,:])
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
        end = start + 0.05 * rays_d[idx_w[i], idx_h[i], :]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c=color, label='Ray Direction')

def plotCameraDirections(df, ax, color, scale_fct):
    # get pose
    x = df["sensor_pose_x"]
    y = df["sensor_pose_y"]
    z = df["sensor_pose_z"]
    pos = np.array([x, y, z])

    # scale position
    pos = scale_fct(pos)


    yaw = df["sensor_pose_yaw"]
    R_yaw_c2w = np.array([[np.cos(yaw), -np.sin(yaw), np.zeros_like(yaw)],
                            [np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)],
                            [np.zeros_like(yaw), np.zeros_like(yaw), np.ones_like(yaw)]])
    
    # Plot points
    ax.scatter(pos[0], pos[1], pos[2], label='Points', color=color)
    
    # Plot directions
    direction = np.matmul(R_yaw_c2w[:,:], np.array([1, 0, 0]))
    ax.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2], length=0.05, normalize=True, color=color)


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

            # # verify ratio height to width
            # dH = np.linalg.norm(rays_d[0,0,:] - rays_d[-1,0,:])
            # dW = np.linalg.norm(rays_d[0,0,:] - rays_d[0,-1,:])
            # if np.abs(dH/dW - dataset.img_wh[1]/dataset.img_wh[0]) > 0.01:
            #     print(f"ERROR: test_read_meta: Ratio height to width is not correct! {dH/dW} != {dataset.img_wh[1]/dataset.img_wh[0]}")

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

            # # verify opening angle
            # W_mid = dataset.img_wh[0] // 2
            # H_mid = dataset.img_wh[1] // 2
            # alpha_H = np.arccos(np.dot(rays_d[0,W_mid,:], rays_d[-1,W_mid,:]))
            # alpha_W = np.arccos(np.dot(rays_d[H_mid,0,:], rays_d[H_mid,-1,:]))
            # print(f"opening angle height = {np.rad2deg(alpha_H)}")
            # print(f"opening angle width = {np.rad2deg(alpha_W)}")

            color = colour_map["cmap"](colour_map["norm"](i+int(dataset.rays.shape[0]/2)))

            plotCameraFoV(rays_o, rays_d, color, ax)
            # plotCameraRays(rays_o, rays_d, ax, color, show_nb_rays)      
            plotCameraDirections(df_split.iloc[i], ax, color, dataset.scalePosition)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses Visualization')
    # ax.set_aspect('equal', 'box')
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks(np.linspace(-0.5, 0.5, 5))
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks(np.linspace(-0.5, 0.5, 5))
    ax.set_zlim(-0.5, 0.5)
    ax.set_zticks(np.linspace(-0.5, 0.5, 5))
    plt.show()



if __name__ == '__main__':
    test_read_meta()