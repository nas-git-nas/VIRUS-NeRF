#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensors.msg import USS

import cv2 as cv
import numpy as np
import pandas as pd
import os
import shutil
import time
import struct
    
    
class LogUSS():
    def __init__(
        self,
        uss_id:list,
        data_dir:str,
        print_elapse_time:bool=False,
        publish_pointcloud:bool=False,
    ):
        """
        Log data from RealSense Node
        Args:
            uss_id: either USS1, USS2 or USS3; str
            data_dir: path where to save data; str
            print_elapse_time: wheter to print ellapse time of callback; bool
            publish_pointcloud: whether to publish pointcloud of tof measurements; bool
        """
        self.uss_id = uss_id
        self.uss_dir = os.path.join(data_dir, uss_id)
        self.print_elapse_time = print_elapse_time
        self.publish_pointcloud = publish_pointcloud
        
        # delete last measurement
        if os.path.exists(self.uss_dir):
            shutil.rmtree(self.uss_dir)
        os.mkdir(os.path.join(self.uss_dir))
        
        # data
        self.sequence = []
        self.seconds = []
        self.nano_seconds = []
        self.measurements = []
        
        # ROS
        self.topic_uss = "/" + self.uss_id
        self.subscribe_uss = None
        rospy.init_node('uss_log', anonymous=True)
        
        # pointcloud
        if self.publish_pointcloud:
            fov = 45 # field of view in degrees
            num_pts_in_row = 64
            
            cell_fov = np.deg2rad(fov) / num_pts_in_row
            angle_max = cell_fov * (num_pts_in_row - 1) / 2
            angle_min = - angle_max
            angles = np.linspace(angle_max, angle_min, num_pts_in_row)
            self.angles_y, self.angles_z = np.meshgrid(angles, angles, indexing="xy") # (N, N), (N, N)
            
            self.pub_pointcloud = rospy.Publisher('tof_pointcloud', PointCloud2, queue_size=10)
            
            self.color_floats = {
                "w": struct.unpack('!f', bytes.fromhex('00FFFFFF'))[0],
                "r": struct.unpack('!f', bytes.fromhex('00FF0000'))[0],
                "g": struct.unpack('!f', bytes.fromhex('0000FF00'))[0],
                "b": struct.unpack('!f', bytes.fromhex('000000FF'))[0],
            }
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        self.subscribe_uss = rospy.Subscriber(self.topic_uss, USS, self._callback)
        
        rospy.loginfo(f"LogUSS.subscribed: Subscribe to: {self.uss_id}")
        rospy.spin()
        
    def save(
        self,
    ):
        """
        Save data.
        """
        df = pd.DataFrame(
            {
                "sequence": self.sequence,
                "time": np.array(self.seconds) + 1e-9*np.array(self.nano_seconds),
                "measurements": self._convertMeasurments(),
            },
        )
       
        df.to_csv(
            os.path.join(self.uss_dir, "data.csv"),
            index=False,
        )
        rospy.loginfo(f"LogUSS.save: Save to: {self.uss_dir}")
        
    def _callback(
        self,
        data:USS,
    ):
        """
        Callback for topic.
        Args:
            data: data from USS; USS
        """
        if self.print_elapse_time:
            start = time.time()
        
        # log data
        self.sequence.append(data.header.seq)
        self.seconds.append(data.header.stamp.secs)
        self.nano_seconds.append(data.header.stamp.nsecs)
        self.measurements.append(data.meas)
        
        if self.publish_pointcloud:
            xyz = self._meas2pointcloud(
                meas=data.meas,
            )
            self._publishPointcloud(
                xyz=xyz,
            )
        
        if self.print_elapse_time:
            rospy.loginfo(f"LogRealSense._callback: elapse time: {(time.time()-start):.3f}s")
            
    def _convertMeasurments(
        self,
    ):
        """
        Convert measurements from time (in us) to distance (in m).
        Every 50ms correspond to 1cm and measurements over 50000us are invalid.
        Returns:
            meas: measurments in meters, np.array of floats
        """
        meas = np.array(self.measurements, dtype=float)
        meas[meas >= 50000] = np.nan
        return meas / 5000
    
    def _meas2pointcloud(
        self,
        meas:float,
    ):
        """
        Converting depths into 3D point cloud.
        Assuming x points into the viewing direction and z upwards.
        Define dim1=z-axis and dim2=y-axis going from larger to smaller values.
        Args:
            meas: depth measurement; float
        Returns:
            xyz: point cloud; np.array (8, 8, 3)
        """
        if meas >= 50000:
            meas = 0
        depth = meas / 5000
        
        x = np.cos(self.angles_y) * np.cos(self.angles_z) * depth # (N, N)
        y = np.sin(self.angles_y) * depth # (N, N)
        z = np.sin(self.angles_z) * depth # (N, N)
        xyz = np.concatenate((x[:,:,None], y[:,:,None], z[:,:,None]), axis=2)
        return xyz
    
    def _publishPointcloud(
        self,
        xyz:np.array,
    ):
        """
        Publish pointcloud as Pointcloud2 ROS message.
        Args:
            xyz: pointcloud; np.array (N, N, 3)
        """
        xyz[xyz==np.NAN] = 0
        xyz = xyz.astype(dtype=np.float32)
        
        rgb = self.color_floats["b"] * np.ones((xyz.shape[0], xyz.shape[1]), dtype=np.float32) # (H, W)
        xyzrgb = np.concatenate((xyz, rgb[:,:,None]), axis=2)
        
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = Header()
        pointcloud_msg.header.frame_id = "RGBD"

        # Define the point fields (attributes)        
        pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        pointcloud_msg.height = xyz.shape[1]
        pointcloud_msg.width = xyz.shape[0]

        # Float occupies 4 bytes. Each point then carries 12 bytes.
        pointcloud_msg.point_step = len(pointcloud_msg.fields) * 4 
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
        pointcloud_msg.is_bigendian = False # assumption
        pointcloud_msg.is_dense = True
        
        pointcloud_msg.data = xyzrgb.tobytes()
        
        self.pub_pointcloud.publish(pointcloud_msg)


def main():
    log = LogUSS(
        uss_id=rospy.get_param("uss_id"),
        data_dir=rospy.get_param("data_dir"),
        print_elapse_time=rospy.get_param("print_elapse_time"),
        publish_pointcloud=rospy.get_param("publish_pointcloud"),
    )
    try:
        log.subscribe()
    finally:
        log.save()

if __name__ == '__main__':
    main()