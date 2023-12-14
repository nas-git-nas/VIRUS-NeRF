import numpy as np
import copy
import rospy

from pcl_tools.pcl_transformer import PCLTransformer

class PCLCoordinator():
    def __init__(
        self,
        source:str,
        target:str,
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
        
        
        # CAM1 -> robot
        if source == "CAM1" and target == "robot":
            self.transform = cam1_robot
            return
        
        # CAM3 -> CAM1
        if source == "CAM3" and target == "CAM1":
            self.transform = cam3_cam1
            return
        
        # robot -> CAM1
        if source == "robot" and target == "CAM1":
            self.transform = cam1_robot.invertTransform()
            return
        
        # CAM1 -> CAM3
        if source == "CAM1" and target == "CAM3":
            self.transform = cam3_cam1.invertTransform()
            return
        
        # CAM3 -> robot or robot -> CAM3
        if (source == "CAM3" and target == "robot") \
            or (source == "robot" and target == "CAM3"):
            T_cam1_robot = cam1_robot.getTransform(
                type="matrix",
            )
            T_cam3_cam1 = cam3_cam1.getTransform(
                type="matrix",
            )
            T_cam3_robot = T_cam1_robot @ T_cam3_cam1
            cam3_robot = PCLTransformer(
                T=T_cam3_robot,
            )
            if source == "CAM3" and target == "robot":
                self.transform = cam3_robot
            else:
                self.transform = cam3_robot.invertTransform()
            return
        
        self.transform = None
        rospy.logerr(f"ERROR: PCLCoordinator.__init__: source={source} and target={target} not implemented")
        
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
        if self.transform is None:
            rospy.logerr(f"ERROR: PCLCoordinator.transformCoordinateSystem: transform is None: : "
                         + "source={self.source} and target={self.target} not implemented")
            return xyz
        
        return self.transform.transformPointcloud(
            xyz=xyz,
        )