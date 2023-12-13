import numpy as np
import copy

from pcl_tools.pcl_transformer import PCLTransformer

class PCLCoordinator():
    def __init__(
        self,
        source:str,
        target:str,
        ) -> None:
        
        # robot -> cam1
        robot_cam1 = PCLTransformer(
            q=[0.5476, 0.3955, 0.4688, 0.5692],
            t=[0.36199, -0.15161, -0.15014],
        )
        if source == "robot" and target == "cam1":
            self.transform = robot_cam1
            return
        
        # cam1 -> cam2
        cam1_cam2 = PCLTransformer(
            q=[-0.00600086, -0.25003579, -0.01240178, -0.96813857],
            t=[0.27075537, 0.00205705, -0.07670919],
        )
        if source == "cam1" and target == "robot":
            self.transform = robot_cam1.invertTransform()
            return
        
        # cam1 -> robot
        if source == "cam1" and target == "cam2":
            cam1_robot = copy.deepcopy(robot_cam1)
            cam1_robot = cam1_robot.invertTransform()
            self.transform = cam1_robot
            return
        
        # cam2 -> cam1
        if source == "cam2" and target == "cam1":
            cam2_cam1 = copy.deepcopy(cam1_cam2)
            cam2_cam1 = cam2_cam1.invertTransform()
            self.transform = cam2_cam1
            return
        
        # robot -> cam2
        if source == "robot" and target == "cam2":
            T_robot_cam1 = robot_cam1.getTransform()
            T_cam1_cam2 = cam1_cam2.getTransform()
            T_robot_cam2 = T_cam1_cam2 @ T_robot_cam1
            T_robot_cam2 = cam1_cam2.getTransform() @ T_robot_cam1
            self.transform = PCLTransformer(
                T=T_robot_cam2,
            )
            return
        
        # cam2 -> robot
        if source == "cam2" and target == "robot":
            T_cam2_robot = copy.deepcopy(T_robot_cam2)
            T_cam2_robot = T_cam2_robot.invertTransform()
            self.transform = T_cam2_robot
            return
        
        print(f"ERROR: PCLCoordinator.__init__: source={source} and target={target} not implemented")
        
    def transformCoordinateSystem(
        self,
        xyz:np.array,
    ):
        """
        Transform pointcloud from source to target coordinate system.
        Args:
            xyz: pointcloud; np.array (N,3)
        Returns:
            xyz: transformed pointcloud; np.array (N,3)
        """
        return self.transform.transformPointcloud(
            xyz=xyz,
        )