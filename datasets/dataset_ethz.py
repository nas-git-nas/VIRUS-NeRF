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
from datasets.scene_ethz import SceneETHZ

from ROS1.src.sensors.src.pcl_tools.pcl_transformer import PCLTransformer
from ROS1.src.sensors.src.pcl_tools.pcl_creator import PCLCreatorUSS, PCLCreatorToF


class DatasetETHZ(DatasetBase):

    def __init__(
        self, 
        args:Args, 
        split:str='train',
        scene:SceneETHZ=None,
    ):

        super().__init__(
            args=args, 
            split=split
        )

        dataset_dir = self.args.ethz.dataset_dir
        data_dir = os.path.join(dataset_dir, self.args.ethz.room)
        cam_ids = [
            "CAM1", 
            "CAM3",
        ]

        # load scene
        self.scene = scene
        if scene is None:
            self.scene = SceneETHZ(
                args=self.args
            )

        # split dataset
        split_mask = self.splitDataset(
            data_dir=data_dir,
            split=split,
        )

        # load camera intrinsics
        img_wh, K_dict, directions_dict = self.readIntrinsics(
            dataset_dir=dataset_dir,
            data_dir=data_dir,
            cam_ids=cam_ids,
        )

        # load samples
        poses, rgbs, depths_dict, sensors_dict, stack_ids = self.readMetas(
            data_dir=data_dir,
            cam_ids=self.args.ethz.cam_ids,
            img_wh=img_wh,
            split_mask=split_mask,
        )

        # if self.args.dataset.keep_pixels_in_angle_range != "all":
        #     rays, directions, depths, img_wh = self.reduceImgHeight(
        #         rays=rays,
        #         directions=directions,
        #         depths=depths,
        #         img_wh=img_wh,
        #         angle_min_max=self.args.dataset.keep_pixels_in_angle_range,
        #     )

        self.img_wh = img_wh
        # self.Ks = Ks
        self.poses = poses
        self.directions_dict = directions_dict
        self.rgbs = rgbs
        self.depths_dict = depths_dict
        self.sensors_dict = sensors_dict
        self.stack_ids = stack_ids
        self.times = None

        # TODO: move to base class
        self.sampler = Sampler(
            args=args,
            dataset_len=len(self),
            img_wh=self.img_wh,
            seed=args.seed,
            sensors_dict=self.sensors_dict,
            fct_getValidDepthMask=self.getValidDepthMask,
        )

    # def getIdxFromSensorName(self, df, sensor_name):
    #     """
    #     Get the indices of the dataset that belong to a particular sensor.
    #     Args:
    #         df: robot@home dataframe, pandas df
    #         sensor_name: name of the sensor, str
    #     Returns:
    #         idxs: indices of the dataset that belong to the sensor
    #     """
    #     sensor_id = self.rh.name2id(sensor_name, "s")
    #     mask = np.array(df["sensor_id"] == sensor_id, dtype=bool)
    #     idxs = np.where(mask)[0]
    #     return idxs

    def splitDataset(
        self,
        data_dir:str,
        split:str,
    ):
        """
        Split the dataset into train and test sets.
        Args:
            data_dir: path to data directory; str
            split: split type; str
        Returns:
            split: split of dataset; bool array of shape (N,)
        """
        path_description = os.path.join(data_dir, 'split', 'split_description.csv')
        path_split = os.path.join(data_dir, 'split', 'split.csv')
        split_ratio = self.args.dataset.split_ratio

        # verify consistendy of dataset length
        N = self._verifyDatasetLength(
            data_dir=data_dir,
        )

        # load split if it exists already
        df_description = None
        if os.path.exists(path_description) and os.path.exists(path_split):    
            df_description = pd.read_csv(
                filepath_or_buffer=path_description,
                dtype={'info':str,'train':float, 'val':float, 'test':float},
            )
        
            # split ratio must be the same as in description (last split)
            if df_description['train'].values[0]==split_ratio['train'] \
                and df_description['val'].values[0]==split_ratio['val'] \
                and df_description['test'].values[0]==split_ratio['test']:

                # load split and merge with df
                df_split = pd.read_csv(
                    filepath_or_buffer=path_split,
                    dtype={'split':str},
                )

                # verify if split has same length as dataset
                if df_split.shape[0] == N:
                    return (df_split["split"].values == split)
                
        # verify that split ratio is correct
        if split_ratio['train'] + split_ratio['val'] + split_ratio['test'] != 1.0:
            self.args.logger.error(f"split ratios do not sum up to 1.0")

        # create new split
        N_train = int(split_ratio['train']*N)
        N_val = int(split_ratio['val']*N)

        rand_idxs = self.args.rng.permutation(N)
        train_idxs = rand_idxs[:N_train]
        val_idxs = rand_idxs[N_train:N_train+N_val]

        split_arr = np.full((N,), "test", dtype=str)
        split_arr[train_idxs] = "train"
        split_arr[val_idxs] = "val"

        # save split and description
        pd.DataFrame(
            data=split_arr,
            columns=["split"],
        ).to_csv(
            filepath_or_buffer=path_split,
            index=False,
        )
        pd.DataFrame(
            data={
                'train':split_ratio['train'], 
                'val':split_ratio['val'], 
                'test':split_ratio['test'], 
                'info':"This file contains the split ratios for this dataset. "
            },
            index=[0],
        ).to_csv(
            filepath_or_buffer=path_description,
            index=False,
        )

        return (split_arr == split)


    def readIntrinsics(
        self,
        dataset_dir:str,
        data_dir:str,
        cam_ids:list,
    ):
        """
        Read camera intrinsics from the dataset.
        Args:
            dataset_dir: path to dataset directory; str
            data_dir: path to data directory; str
            cam_ids: list of camera ids; list of str
        Returns:
            img_wh: tuple of image width and height
            K_dict: camera intrinsic matrix dictionary; dict oftensor of shape (3, 3)
            directions_dict: ray directions dictionary; dict of tensor of shape (H*W, 3)
        """
        # get image width and height
        img_path = os.path.join(data_dir, 'measurements/CAM1_color_image_raw', 'img0.png')
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        h, w, _ = img.shape
        img_wh = (w, h)

        # get camera intrinsics
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(dataset_dir, 'camera_intrinsics.csv'),
            dtype={'cam_id': str, 'fx': np.float64, 'fy': np.float64, 'cx': np.float64, 'cy': np.float64},
        )
        K_dict = {}
        for cam_id in cam_ids:
            df_cam = df[df["cam_id"]==cam_id]
            K_dict[cam_id] = np.array([[df_cam['fx'].values[0], 0.0, df_cam['cx'].values[0]],
                                  [0.0, df_cam['fy'].values[0], df_cam['cy'].values[0]], 
                                  [0.0, 0.0, 1.0]])

        # get ray directions
        directions_dict = {}
        for cam_id in cam_ids:
            directions_dict[cam_id] = get_ray_directions(h, w, K_dict[cam_id]) # (H*W, 3)

        # convert numpy arrays to tensors
        for cam_id in cam_ids:
            K_dict[cam_id] = torch.tensor(K_dict[cam_id], dtype=torch.float32, requires_grad=False, device=self.args.device)
            directions_dict[cam_id] = torch.tensor(directions_dict[cam_id], dtype=torch.float32, requires_grad=False, device=self.args.device)

        return img_wh, K_dict, directions_dict

    def readMetas(
        self,
        data_dir:str,
        cam_ids:list,
        img_wh:tuple,
        split_mask:np.array,
    ):
        """
        Read all samples from the dataset.
        Args:
            data_dir: path to data directory; str
            cam_ids: list of camera ids; list of str
            img_wh: image width and height; tuple of ints
            split_mask: mask of split; bool array of shape (N,)
        Returns:
            poses: camera poses; array of shape (N_images, 3, 4)
            rgbs: ray origins; array of shape (N_images, H*W, 3)
            depths_dict: dictionary of depth samples; dict of { sensor type: array of shape (N_images, H*W) }
            sensors_dict: dictionary of sensor models; dict of { sensor: sensor model }
            stack_ids: stack identity number of sample; array of shape (N_images,)
        """
        poses, stack_ids = self._readPoses(
            data_dir=data_dir,
            cam_ids=cam_ids,
            split_mask=split_mask,
        ) # (N, 3, 4),  (N,)
        poses = self._convertPoses(
            poses=poses,
        ) # (N, 3, 4)

        rgbs, rgbs_stack_ids = self._readColorImgs(
            data_dir=data_dir,
            cam_ids=cam_ids,
            img_wh=img_wh,
            split_mask=split_mask,
        ) # (N, H*W, 3), (N,)
        if self.args.model.debug_mode and not np.all(stack_ids == rgbs_stack_ids):
            self.args.logger.error(f"DatasetETHZ::read_meta: stack ids do not match")

        depths_dict = {}
        sensors_dict = {}
        if "USS" in self.args.training.sensors:
            uss_meass, uss_stack_ids = self._readUSS(
                data_dir=data_dir,
                cam_ids=cam_ids,
                split_mask=split_mask,
            ) # (N,), (N,)
            if self.args.model.debug_mode and not np.all(stack_ids == uss_stack_ids):
                self.args.logger.error(f"DatasetETHZ::read_meta: uss_stack_ids ids do not match")

            uss_depths, uss_sensors_dict = self._convertUSS(
                meass=uss_meass,
                stack_ids=uss_stack_ids,
                img_wh=img_wh,
            ) # (N, H*W), dict { cam_id : USSModel }

            depths_dict["USS"] = uss_depths
            sensors_dict.update(uss_sensors_dict)

        if "ToF" in self.args.training.sensors:
            tof_meass, tof_meas_stds, tof_stack_ids = self._readToF(
                data_dir=data_dir,
                cam_ids=cam_ids,
                split_mask=split_mask,
            ) # (N, 64), (N, 64), (N,)
            if self.args.model.debug_mode and not np.all(stack_ids == tof_stack_ids):
                self.args.logger.error(f"DatasetETHZ::read_meta: tof_stack_ids ids do not match")

            tof_depths, tof_stds, tof_sensors_dict = self._convertToF(
                meass=tof_meass,
                meas_stds=tof_meas_stds,
                stack_ids=tof_stack_ids,
                img_wh=img_wh,
            ) # (N, H*W), (N, H*W), dict { cam_id : ToFModel }

            depths_dict["ToF"] = tof_depths
            sensors_dict.update(tof_sensors_dict)

        return poses, rgbs, depths_dict, sensors_dict, stack_ids
    
    def _readPoses(
        self,
        data_dir:str,
        cam_ids:list,
        split_mask:np.array,
    ):
        """
        Read poses from the dataset for each camera.
        Args:
            cam_ids: list of camera ids; list of str
            data_dir: path to data directory; str
            split_mask: mask of split; bool array of shape (N_all_splits,)
        Returns:
            poses: camera poses; array of shape (N, 3, 4)
            stack_ids: stack identity number of sample; array of shape (N,)
        """
        poses = np.zeros((0, 3, 4))
        stack_ids = np.zeros((0))
        for cam_id in cam_ids:
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(data_dir, 'poses', 'poses_sync'+cam_id[-1]+'cam_robot.csv'),
                dtype=np.float64,
            )

            pose = np.zeros((np.sum(split_mask), 3, 4))
            for i in np.arange(df.shape[0])[split_mask]:
                trans = PCLTransformer(
                    t=[df["x"][i], df["y"][i], df["z"][i]],
                    q=[df["qx"][i], df["qy"][i], df["qz"][i], df["qw"][i]],
                )
                pose[i] = trans.getTransform(
                    type="matrix",
                )[:3,:] # (3, 4)

            poses = np.concatenate((poses, pose), axis=0) # (N, 3, 4)
            stack_ids = np.concatenate((stack_ids, np.ones((pose.shape[0]))*int(cam_id[-1])), axis=0) # (N,)

        return poses, stack_ids
    
    def _readColorImgs(
        self,
        data_dir:str,
        cam_ids:list,
        img_wh:tuple,
        split_mask:np.array,
    ):
        """
        Read color images from the dataset for each camera.
        Args:
            cam_ids: list of camera ids; list of str
            data_dir: path to data directory; str
            img_wh: image width and height; tuple of ints
            split_mask: mask of split; bool array of shape (N_all_splits,)
        Returns:
            rgbs_dict: color images; array of shape (N, H*W, 3)
            stack_ids: stack identity number of sample; array of shape (N,)
        """
        W, H = img_wh

        rgbs = np.zeros((0, H*W, 3))
        stack_ids = np.zeros((0))
        for cam_id in cam_ids:
            rgb_path = os.path.join(data_dir, 'measurements/'+cam_id+'_color_image_raw') 
            rgb_files = ['img'+str(i)+'.png' for i in range(split_mask.shape[0])]
            rgb_files = rgb_files[split_mask]

            rgbs_temp = np.zeros((len(rgb_files), H*W, 3))
            for i, f in enumerate(rgb_files):
                rgb_file = os.path.join(rgb_path, f)
                rgb = cv.imread(rgb_file, cv.IMREAD_COLOR) # (H, W, 3)
                rgbs_temp[i] = rgb.reshape(H*W, 3) # (H*W, 3)

            rgbs = np.concatenate((rgbs, rgbs_temp), axis=0) # (N, H*W, 3)
            stack_ids = np.concatenate((stack_ids, np.ones((rgbs_temp.shape[0]))*int(cam_id[-1])), axis=0) # (N,)

        return rgbs, stack_ids

    def _readDepthImgs(
        self,
        data_dir:str,
        cam_ids:list,
        img_wh:tuple,
        split_mask:np.array,
    ):
        """
        Read depth images from the dataset for each camera.
        Args:
            cam_ids: list of camera ids; list of str
            data_dir: path to data directory; str
            img_wh: image width and height; tuple of ints
            split_mask: mask of split; bool array of shape (N_all_splits,)
        Returns:
            depths: depth images; array of shape (N, H*W)
            stack_ids: stack identity number of sample; array of shape (N,)
        """
        W, H = img_wh

        depths = np.zeros((0, H*W))
        stack_ids = np.zeros((0))
        for cam_id in cam_ids:
            depth_path = os.path.join(data_dir, 'measurements/'+cam_id+'_aligned_depth_to_color_image_raw')
            depth_files = ['img'+str(i)+'.png' for i in range(split_mask.shape[0])]
            depth_files = depth_files[split_mask]

            depths_temp = np.zeros((len(depth_files), H*W))
            for i, f in enumerate(depth_files):
                depth_file = os.path.join(depth_path, f)
                depth = cv.imread(depth_file, cv.IMREAD_UNCHANGED)
                depths_temp[i] = depth[:,:,0].flatten() # (H*W), keep only one color channel

                # verify depth image
                if self.args.model.debug_mode:
                    if not (np.allclose(depth[:,:,0], depth[:,:,1]) and np.allclose(depth[:,:,0], depth[:,:,2])):
                        self.args.logger.error(f"robot_at_home.py: read_meta: depth image has more than one channel")

            depths = np.concatenate((depths, depths_temp), axis=0) # (N, H*W)
            stack_ids = np.concatenate((stack_ids, np.ones((depths_temp.shape[0]))*int(cam_id[-1])), axis=0) # (N,)

        return # TODO: convert depth to radius and to cube coordinate system  
    
    def _readUSS(
        self,
        data_dir:str,
        cam_ids:list,
        split_mask:np.array,
    ):
        """
        Read USS measurements from the dataset for each camera.
        Args:
            cam_ids: list of camera ids; list of str
            data_dir: path to data directory; str
            split_mask: mask of split; bool array of shape (N_all_splits,)
        Returns:
            meass: USS measurements; array of shape (N_images,)
            stack_ids: stack identity number of sample; array of shape (N_images,)
        """
        meass = np.zeros((0))
        stack_ids = np.zeros((0))
        for cam_id in cam_ids:
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(data_dir, 'measurements/USS'+cam_id[-1]+'.csv'),
                dtype=np.float32,
            )
            meass_temp = df["meas"].to_numpy()
            meass_temp = meass_temp[split_mask]

            meass = np.concatenate((meass, meass_temp), axis=0) # (N,)
            stack_ids = np.concatenate((stack_ids, np.ones((meass_temp.shape[0]))*int(cam_id[-1])), axis=0) # (N,)

        return meass, stack_ids
    
    def _readToF(
        self,
        data_dir:str,
        cam_ids:list,
        split_mask:np.array,
    ):
        """
        Read Tof measurements from the dataset for each camera.
        Args:
            cam_ids: list of camera ids; list of str
            data_dir: path to data directory; str
            split_mask: mask of split; bool array of shape (N_all_splits,)
        Returns:
            meass: USS measurements; array of shape (N_images, 64)
            meas_stds: USS measurements; array of shape (N_images, 64)
            stack_ids: stack ids; array of shape (N_images,)
        """
        meass = np.zeros((0, 64))
        meas_stds = np.zeros((0, 64))
        stack_ids = np.zeros((0))
        for cam_id in cam_ids:
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(data_dir, 'measurements/USS'+cam_id[-1]+'.csv'),
                dtype=np.float32,
            )

            meass_temp = np.zeros((df.shape[0], 64))
            stds = np.zeros((df.shape[0], 64))
            for i in range(df.shape[0]):
                meass_temp[:,i] = df["meas_"+str(i)].to_numpy()
                stds[:,i] = df["std_"+str(i)].to_numpy()
            
            meass_temp = meass_temp[split_mask]
            stds = stds[split_mask]

            meass = np.concatenate((meass, meass_temp), axis=0) # (N, 64)
            meas_stds = np.concatenate((meas_stds, stds), axis=0)
            stack_ids = np.concatenate((stack_ids, np.ones((meass_temp.shape[0]))*int(cam_id[-1])), axis=0) # (N,)

        return meass, meas_stds, stack_ids
    
    def _convertPoses(
        self,
        poses:dict,
    ):
        """
        Convert poses to cube coordinates.
        Args:
            poses: camera poses; array of shape (N_images, 3, 4)
        Returns:
            poses: camera poses in cube coordinates; array of shape (N_images, 3, 4)
        """
        # convert positions from world to cube coordinate system
        xyz = poses[:,:,3] # (N, 3)
        xyz = self.scene.w2c(pos=xyz, copy=False) # (N, 3)
        poses[:,:,3] = xyz # (N, 3, 4)
        
        # convert array to tensor
        poses = torch.tensor(
            data=poses,
            dtype=torch.float32,
            requires_grad=False,
            device=self.args.device,
        )
        return poses
    
    def _convertUSS(
        self,
        meass:dict,
        stack_ids:np.array,
        img_wh:tuple,
    ):
        """
        Convert USS measurement to depth in cube coordinates.
        Args:
            meass: dictionary containing USS measurements; array of shape (N_images,)
            stack_ids: stack identity number of sample; array of shape (N_images,)
            img_wh: image width and height; tuple of ints
        Returns:
            depths: converted depths; array of shape (N_images, H*W)
            sensors_dict: dictionary containing sensor models; dict { cam_id : USSModel }
        """
        pcl_creator = PCLCreatorUSS(
                W=1,
                H=1,
            )
        
        # convert USS measurements to depth in meters
        depths = np.zeros((meass.shape[0])) # (N)
        for i, meas in enumerate(meass):
            depths[i] = pcl_creator.meas2depth(
                meas=meas,
            )

        # convert depth in meters to cube coordinates [-0.5, 0.5]
        depths = self.scene.w2c(depths.flatten(), only_scale=True) # (N,)

        sensors_dict = {} 
        for id in np.unique(stack_ids):
            sensor_maks = (stack_ids == id)
            sensor_id = "USS"+str(id)

            # create sensor model
            sensors_dict[sensor_id] = USSModel(
                args=self.args, 
                img_wh=img_wh,
                num_imgs=np.sum(sensor_maks),
            )

            # mask pixels that are outside of the field of view
            depths[sensor_maks] = sensors_dict[sensor_id].convertDepth(
                depths=depths[sensor_maks],
                format="sensor",
            ) # (N, H*W)

        # convert depth to tensor
        depths =  torch.tensor(
            data=depths,
            dtype=torch.float32,
            requires_grad=False,
        )
        return depths, sensors_dict

    def _convertToF(
        self,
        meass:np.array,
        meas_stds:np.array,
        stack_ids:np.array,
        img_wh:tuple,
    ):
        """
        Convert ToF measurement to depth in cube coordinates.
        Args:
            meass: dictionary containing ToF measurements; array of shape (N_images, 64,)
            meas_stds: dictionary containing ToF measurement standard deviations; array of shape (N_images, 64,)
            stack_ids: stack identity number of sample; array of shape (N_images,)
            img_wh: image width and height; tuple of ints
        Returns:
            depths: converted depths; array of shape (N_images, H*W)
            stds: converted standard deviations; array of shape (N_images, H*W)
            sensors_dict: dictionary containing sensor models; dict { cam_id : ToFModel }
        """
        pcl_creator = PCLCreatorToF(
            W=8,
            H=8,
        )
        
        # convert ToF measurements to depth in meters
        depths = np.zeros((meass.shape[0], 8, 8)) # (N, 8, 8)
        stds = np.zeros((meas_stds.shape[0], 8, 8)) # (N, 8, 8)
        for i in range(meass.shape[0]):
            depths[i] = pcl_creator.meas2depth(
                meas=meass[i],
            ) # (8, 8)
            stds[i] = pcl_creator.meas2depth(
                meas=meas_stds[i],
            ) # (8, 8)

        # convert depth in meters to cube coordinates [-0.5, 0.5]
        depths = self.scene.w2c(depths.flatten(), only_scale=True).reshape(-1, 64) # (N, 8*8)
        stds = self.scene.w2c(stds.flatten(), only_scale=True).reshape(-1, 64) # (N, 8*8)

        sensors_dict = {} 
        for id in np.unique(stack_ids):
            sensor_maks = (stack_ids == id)
            sensor_id = "ToF"+str(id)

            # create sensor model
            sensors_dict[sensor_id] = ToFModel(
                args=self.args, 
                img_wh=img_wh,
            )

            # mask pixels that are outside of the field of view
            depths[sensor_maks] = sensors_dict[sensor_id].convertDepth(
                depths=depths[sensor_maks],
                format="sensor",
            ) # (N, H*W)
            # mask pixels that are outside of the field of view
            stds[sensor_maks] = sensors_dict[sensor_id].convertDepth(
                depths=stds[sensor_maks],
                format="sensor",
            ) # (N, H*W)

        # convert depth to tensor
        depths =  torch.tensor(
            data=depths,
            dtype=torch.float32,
            requires_grad=False,
        )
        stds =  torch.tensor(
            data=stds,
            dtype=torch.float32,
            requires_grad=False,
            )
        return depths, stds, sensors_dict
    
    def _verifyDatasetLength(
        self,
        data_dir:str,
    ):
        """
        Verify that the dataset length is the same for all sensors.
        Args:
            data_dir: path to data directory; str
        Returns:
            N: length of dataset; int
        """
        N = None # length of dataset

        df_names = [
            'measurements/USS1.csv',
            'measurements/USS3.csv',
            'measurements/TOF1.csv',
            'measurements/TOF3.csv',
        ]
        for name in df_names:
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(data_dir, name),
                dtype={'time':str, 'meas':np.float32},
            )
            if N is None:
                N = df.shape[0]
            elif N != df.shape[0]:
                self.args.logger.error(f"DatasetETHZ::_verifyDatasetLength: dataset length "
                                       + f"is not the same for all sensors!")
                return None
                
        dir_names = [
            'measurements/CAM1_color_image_raw',
            'measurements/CAM3_color_image_raw',
            'measurements/CAM1_aligned_depth_to_color_image_raw',
            'measurements/CAM3_aligned_depth_to_color_image_raw',
        ]
        for name in dir_names:
            files = os.listdir(os.path.join(data_dir, name))
            if N != len(files):
                self.args.logger.error(f"DatasetETHZ::_verifyDatasetLength: dataset length "
                                       + f"is not the same for all sensors!")
                return None
        return N

    

    

   