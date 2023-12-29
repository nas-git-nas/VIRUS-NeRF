import numpy as np
import copy
try:
    import rospy
    ROSPY_AVAILABLE = True
except:
    ROSPY_AVAILABLE = False
import pandas as pd
import os

try:
    from pcl_tools.pcl_transformer import PCLTransformer
except:
    from ROS1.src.sensors.src.pcl_tools.pcl_transformer import PCLTransformer

class PCLCoordinator():
    def __init__(
        self,
        source:str,
        target:str,
        lookup_table_path:str=None,
        ) -> None:
        self.source = source
        self.target = target
        
        # if a lookup table exist, use it
        self.df_lookup_table = None
        if lookup_table_path is not None:
            self.df_lookup_table = pd.read_csv(
                lookup_table_path,
                dtype=np.float64,
            )
            return
        
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
        
        # CAM3 -> robot
        if source == "CAM3" and target == "robot":
            cam3_robot = cam1_robot.concatTransform(
                add_transform=cam3_cam1,
                apply_first_add_transform=True,
            )
            self.transform = cam3_robot
            return
        
        # robot -> CAM3
        if source == "robot" and target == "CAM3":
            cam3_robot = cam1_robot.concatTransform(
                add_transform=cam3_cam1,
                apply_first_add_transform=True,
            )
            self.transform = cam3_robot.invertTransform()
            return
        
        self.transform = None
        if ROSPY_AVAILABLE:
            rospy.logerr(f"ERROR: PCLCoordinator.__init__: source={source} and target={target} not implemented")
        else:
            print(f"ERROR: PCLCoordinator.__init__: source={source} and target={target} not implemented")
        
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
        if self.df_lookup_table is not None:
            trans = self._lookupTransform(
                time=time,
            )
        else:
            trans = self.transform
        
        return trans.transformPointcloud(
            xyz=xyz,
        )
            
    
    def _lookupTransform(
        self,
        time:float,
    ):
        """
        Lookup dynamic transform.
        Args:
            time: time of pointcloud; float
        Returns:
            trans: transform from camera to map coordinate system; PCLTransformer
        """
        if time is None:
            if ROSPY_AVAILABLE:
                rospy.logerr(f"ERROR: PCLCoordinator._lookupTransform: time={time} not given")
            else:
                print(f"ERROR: PCLCoordinator._lookupTransform: time={time} not given")
        
        mask = np.abs(self.df_lookup_table['time'] - time) < 1e-4
        if np.sum(mask) != 1:
            error = np.abs(self.df_lookup_table['time'] - time)
            error_min = np.min(error)
            if ROSPY_AVAILABLE:
                rospy.logerr(f"ERROR: PCLCoordinator._lookupTransform: time={time} not found in lookup table: np.sum(mask)={np.sum(mask)}")
                rospy.logerr(f"ERROR: PCLCoordinator._lookupTransform: min error={error_min}")
            else:
                print(f"ERROR: PCLCoordinator._lookupTransform: time={time} not found in lookup table: np.sum(mask)={np.sum(mask)}")
                print(f"ERROR: PCLCoordinator._lookupTransform: min error={error_min}")
            
        row = self.df_lookup_table.loc[mask]        
        trans = PCLTransformer(
            q=row[['qx', 'qy', 'qz', 'qw']].values[0],
            t=row[['x', 'y', 'z']].values[0],
        )
        return trans
    
    
    
def createLookupTables(
    data_dir:str,
):
    """
    Create lookup table for dynamic transforms.
    Args:
        data_dir: path to data directory; str
    """
    lookupTable(
        data_dir=data_dir,
        stack_id=1,
    )
    lookupTable(
        data_dir=data_dir,
        stack_id=3,
    )
    
    
def lookupTable(
    data_dir:str,
    stack_id:int,
):
    """
    Create lookup table for dynamic transforms.
    Args:
        data_dir: path to data directory; str
    """
    # read robot poses
    df_poses_robot = pd.read_csv(
        os.path.join(data_dir, 'poses', 'poses_sync'+str(stack_id)+'.csv'),
        dtype=np.float64,
    )
    
    df_poses_cam = pd.DataFrame(
        columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
        dtype=np.float64,
    )
    
    coord = PCLCoordinator(
        source="CAM"+str(stack_id),
        target="robot",
    )
    
    for i in range(len(df_poses_robot)):
        row = df_poses_robot.iloc[i]
        
        
        # lookup transform from robot to map
        robot_map = PCLTransformer(
            q=row[['qx', 'qy', 'qz', 'qw']].values,
            t=row[['x', 'y', 'z']].values,
        )
        
        # calculate transform from CAM to robot
        cam_robot = copy.deepcopy(coord.transform)
        cam_map = cam_robot.concatTransform(
            add_transform=robot_map,
            apply_first_add_transform=False,
        )
        
        # add transform from CAM to map to dataframe
        q, t = cam_map.getTransform(type="quaternion")
        time = np.array([row['time']])
        arr = np.concatenate((time, t, q))
        df_poses_cam = df_poses_cam.append(
            pd.DataFrame(
                data=np.concatenate((time, t, q)).reshape(1,-1),
                columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
            ),
            ignore_index=True,
        )
    
    # save lookup table
    df_poses_cam.to_csv(
        path_or_buf=os.path.join(data_dir, 'poses', 'poses_sync'+str(stack_id)+'_cam_robot.csv'),
        index=False,
    )
    
def main():
    data_dir = "/home/spadmin/catkin_ws_ngp/data/office_2"
    createLookupTables(
        data_dir=data_dir,
    )
    
if __name__ == '__main__':
    main()