import numpy as np
import copy
import rospy
import pandas as pd
import os

from pcl_tools.pcl_transformer import PCLTransformer

class PCLCoordinator():
    def __init__(
        self,
        source:str,
        target:str,
        data_dir:str=None,
        ) -> None:
        self.source = source
        self.target = target
        
        # Fixed: CAM1 -> robot
        cam1_robot = PCLTransformer(
            q=[0.5476, 0.3955, 0.4688, 0.5692],
            t=[0.36199, -0.15161, -0.15014],
        )
        
        # Fixed: CAM3 -> cam1
        cam3_cam1 = PCLTransformer(
            q=[-0.00600086, -0.25003579, -0.01240178, -0.96813857],
            t=[0.27075537, 0.00205705, -0.07670919],
        )
        
        if source == "map" or target == "map":
            if source == "CAM1" or target == "CAM1":
                self.df_poses = pd.read_csv(
                    os.path.join(data_dir, 'poses_sync1.csv'),
                )
            elif source == "CAM3" or target == "CAM3":
                self.df_poses = pd.read_csv(
                    os.path.join(data_dir, 'poses_sync3.csv'),
                )
            else:
                rospy.logerr(f"ERROR: PCLCoordinator.__init__: source={source} and target={target} not implemented")
        
        # CAM1 -> robot
        if source == "CAM1" and (target == "robot" or target == "map"):
            self.transform = cam1_robot
            return
        
        # CAM3 -> CAM1
        if source == "CAM3" and target == "CAM1":
            self.transform = cam3_cam1
            return
        
        # robot -> CAM1
        if (source == "robot" or source == "map") and target == "CAM1":
            self.transform = cam1_robot.invertTransform()
            return
        
        # CAM1 -> CAM3
        if source == "CAM1" and target == "CAM3":
            self.transform = cam3_cam1.invertTransform()
            return
        
        # CAM3 -> robot
        if source == "CAM3" and (target == "robot" or target == "map"):
            cam3_robot = self._calcCam3Robot(
                cam1_robot=cam1_robot,
                cam3_cam1=cam3_cam1,
            )
            self.transform = cam3_robot
            return
        
        # robot -> CAM3
        if (source == "robot" or source == "map") and target == "CAM3":
            cam3_robot = self._calcCam3Robot(
                cam1_robot=cam1_robot,
                cam3_cam1=cam3_cam1,
            )
            self.transform = cam3_robot.invertTransform()
            return
        
        self.transform = None
        rospy.logerr(f"ERROR: PCLCoordinator.__init__: source={source} and target={target} not implemented")
        
    def transformCoordinateSystem(
        self,
        xyz:np.array,
        time:float=None,
    ):
        """
        Transform pointcloud from source to target coordinate system.
        Args:
            xyz: pointcloud; np.array (N,3)
            time: time of pointcloud; float
        Returns:
            xyz: transformed pointcloud; np.array (N,3)
        """            
        # return static transformed pointcloud
        if self.source != "map" and self.target != "map":
            return self.transform.transformPointcloud(
                xyz=xyz,
            )
            
        robot_map = self._lookupRobotMapTransform(
            time=time,
        )
        trans = copy.deepcopy(self.transform)
        
        if self.target == "map":
            trans = trans.concatTransform(
                add_transform=robot_map,
                apply_first_add_transform=False,
            )
        elif self.source == "map":
            trans = trans.concatTransform(
                add_transform=robot_map.invertTransform(),
                apply_first_add_transform=True,
            )
        else:
            rospy.logerr(f"ERROR: PCLCoordinator.transformCoordinateSystem: "\
                         + f"source={self.source} and target={self.target} not implemented")
        
        return trans.transformPointcloud(
            xyz=xyz,
        )
    
    def _lookupRobotMapTransform(
        self,
        time:float,
    ):
        """
        Lookup dynamic transform.
        Args:
            time: time of pointcloud; float
        Returns:
            robot_map: transform from robot to map coordinate system; PCLTransformer
        """
        row = self.df_sync.loc[self.df_sync['time'] == time]
        
        robot_map = PCLTransformer(
            q=row[['qx', 'qy', 'qz', 'qw']].values[0],
            t=row[['x', 'y', 'z']].values[0],
        )
        return robot_map
        
    def _calcCam3Robot(
        self,
        cam1_robot:PCLTransformer,
        cam3_cam1:PCLTransformer,
    ):
        """
        Calculate transformation from CAM3 to robot coordinate system.
        Args:
            cam1_robot: transformation from CAM1 to robot coordinate system; PCLTransformer
            cam3_cam1: transformation from CAM3 to CAM1 coordinate system; PCLTransformer
        Returns:
            cam3_robot: transformation from CAM3 to robot coordinate system; PCLTransformer
        """
        cam3_robot = cam1_robot.concatTransform(
            add_transform=cam3_cam1,
            apply_first_add_transform=True,
        )
        
        # T_cam1_robot = cam1_robot.getTransform(
        #     type="matrix",
        # )
        # T_cam3_cam1 = cam3_cam1.getTransform(
        #     type="matrix",
        # )
        
        # T_cam3_robot = T_cam1_robot @ T_cam3_cam1
        # cam3_robot = PCLTransformer(
        #     T=T_cam3_robot,
        # )
        return cam3_robot