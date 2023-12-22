import numpy as np
import os
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from robotathome import RobotAtHome
from args.args import Args
from datasets.scene_base import SceneBase
from ROS1.src.sensors.src.pcl_tools.pcl_loader import PCLLoader

class SceneETHZ(SceneBase):
    def __init__(
        self,
        args:Args,
        data_dir:str,
    ):
        """
        Class to handle ETHZ-dataset scenes. The scene's point cloud serves as ground truth.
        Args:
            args: arguments; Args object
            data_dir: path to data directory; str
        """ 
        self.data_dir = data_dir
        self.pcl_dir = "maps"

        if args.ethz.room == "office_2":
            self.xyz_min = np.array([-2.0, -2.0, -0.4])
            self.xyz_max = np.array([7.0, 10.0, 1.2])
        elif args.ethz.room == "medium_2":
            self.xyz_min = np.array([-3.0, -3.0, -0.4])
            self.xyz_max = np.array([15.0, 9.0, 1.2])

        SceneBase.__init__(
            self, 
            args=args,
        )
        
    def _loadPointCloud(
        self,
    ):
        """
        Load scene from ETHZ dataset
        Returns:
            point cloud of scene; numpy array (N, x y z)
        """
        pcl_loader = PCLLoader(
            data_dir=self.data_dir,
            pcl_dir=self.pcl_dir,
        )
        return pcl_loader.loadLatestPCL()
    
    def _defineParams(self):
        """
        Calculate world (in meters) to cube ([cube_min,cube_max]**3) transformation parameters.
        Enlarge the cube with scale_margin s.t. all points are sure to be inside the cube.
        """
        # get scene shift and scale
        xyz_min = self.xyz_min
        xyz_max = self.xyz_max

        shift = (xyz_max + xyz_min) / 2 # TODO: move to base class
        scale = (xyz_max - xyz_min).max() * self.w2c_params["scale_margin"] \
                / (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]) 

        # set world to cube transformation parameters
        self.w2c_params["defined"] = True
        self.w2c_params["shift"] = shift
        self.w2c_params["scale"] = scale