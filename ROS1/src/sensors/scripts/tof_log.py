#!/usr/bin/env python
import rospy
from sensors.msg import TOF
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import cv2 as cv
import numpy as np
import pandas as pd
import os
import shutil
import time
import struct
    
    
class LogToF():
    def __init__(
        self,
        tof_id:list,
        data_dir:str,
        print_elapse_time:bool=False,
        publish_pointcloud:bool=False,
    ):
        """
        Log data from RealSense Node
        Args:
            camera_id: either TOF1, TOF2 or TOF3; str
            data_dir: path where to save data; str
            print_elapse_time: wheter to print ellapse time of callback; bool
            publish_pointcloud: whether to publish pointcloud of tof measurements; bool
        """
        self.tof_id = tof_id
        self.tof_dir = os.path.join(data_dir, tof_id)
        self.print_elapse_time = print_elapse_time
        self.publish_pointcloud = publish_pointcloud
        
        # delete-data of last measurement
        if os.path.exists(self.tof_dir):
            shutil.rmtree(self.tof_dir)
        os.mkdir(self.tof_dir)
        os.mkdir(os.path.join(self.tof_dir, "imgs"))
        
        # data
        self.sequence = []
        self.seconds = []
        self.nano_seconds = []
        self.meas = []
        self.stds = []
        
        # ROS
        self.topic_tof = "/" + self.tof_id
        self.subscribe_tof = None
        rospy.init_node('tof_log', anonymous=True)
        
        # pointcloud
        if self.publish_pointcloud:
            fov = 45 # field of view in degrees
            num_cells_in_row = 8
            
            cell_fov = np.deg2rad(fov) / num_cells_in_row
            angle_max = cell_fov * (num_cells_in_row - 1) / 2
            angle_min = - angle_max
            angles = np.linspace(angle_max, angle_min, num_cells_in_row)
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
        self.subscribe_tof = rospy.Subscriber(self.topic_tof, TOF, self._callback)
        
        rospy.loginfo(f"LogToF.subscribe: Subscribed to: {self.tof_id}")
        rospy.spin()
        
    def save(
        self,
    ):
        """
        Save data.
        """      
        # create pandas dataframe with data
        df = pd.DataFrame(
            data={
                "sequence": self.sequence,
                "time": np.array(self.seconds) + 1e-9*np.array(self.nano_seconds),
            },
        )
       
        # save data
        df.to_csv(
            os.path.join(self.tof_dir, "meta_data.csv"),
            index=False,
        )
        rospy.loginfo(f"LogToF.save: Save to: {self.tof_dir}")
        
    def _callback(
        self,
        data:TOF,
    ):
        """
        Callback for topic.
        Args:
            data: measurements from ToF
        """
        if self.print_elapse_time:
            start = time.time()
        
        # log data
        self.sequence.append(data.header.seq)
        self.seconds.append(data.header.stamp.secs)
        self.nano_seconds.append(data.header.stamp.nsecs)
        
        # save img
        status = cv.imwrite(
            filename=os.path.join(self.tof_dir, "imgs", f"meas{self.sequence[-1]}.png"), 
            img=np.array(data.meas, dtype=np.uint16).reshape(8, 8),
        )
        if not status:
            rospy.logwarn(f"LogToF._callback: meas save status: {status}")
        status = cv.imwrite(
            filename=os.path.join(self.tof_dir, "imgs", f"stds{self.sequence[-1]}.png"), 
            img=np.array(data.stds, dtype=np.uint16).reshape(8, 8),
        )
        if not status:
            rospy.logwarn(f"LogToF._callback: stds save status: {status}")
        
        if self.publish_pointcloud:
            meas = self._convertMeasurments(
                meas=data.meas,
            )
            meas = self._transformMeasurments(
                meas=meas,
            )
            xyz = self._depth2pointcloud(
                depths=meas,
            )
            self._publishPointcloud(
                xyz=xyz,
            )
        
        if self.print_elapse_time:
            rospy.loginfo(f"LogRealSense._callback: elapse time: {(time.time()-start):.6f}s")
            
    def _convertMeasurments(
        self,
        meas,
    ):
        """
        Convert measurements from distance in mm to distance in m.
        Invalid measurements are equal to 0.
        Args:
            meas: N measurements as tubles; tuble of floats (64)
        Returns:
            meas: measurments in meters; np.array of floats (8, 8)
        """
        meas = np.array(meas, dtype=float).reshape(8, 8)
        
        meas[meas == 0] = np.nan
        return meas / 1000
    
    def _transformMeasurments(
        self,
        meas,
    ):
        """
        Convert measurements from distance in mm to distance in m.
        Measurements equal to 0 are invalid.
        Args:
            meas: measurements not transform; np.array of floats (N, 8, 8)
        Returns:
            meas: measurments transformed; np.array of floats (N, 8, 8)
        """
        meas = meas[:, ::-1]
        return meas.T
    
    def _depth2pointcloud(
        self,
        depths:np.array,
    ):
        """
        Converting depths into 3D point cloud.
        Assuming x points into the viewing direction and z upwards.
        Define dim1=z-axis and dim2=y-axis going from larger to smaller values.
        Args:
            depths: depth measurements; np.array (8, 8)
        Returns:
            xyz: point cloud; np.array (8, 8, 3)
        """
        depths[depths>1.0] = 0
        
        x = np.cos(self.angles_y) * np.cos(self.angles_z) * depths # (N, N)
        y = np.sin(self.angles_y) * depths # (N, N)
        z = np.sin(self.angles_z) * depths # (N, N)
        xyz = np.concatenate((x[:,:,None], y[:,:,None], z[:,:,None]), axis=2)
        return xyz
    
    def _publishPointcloud(
        self,
        xyz:np.array,
    ):
        """
        Publish pointcloud as Pointcloud2 ROS message.
        Args:
            xyz: pointcloud; np.array (8, 8, 3)
        """
        xyz[xyz==np.NAN] = 0
        xyz = xyz.astype(dtype=np.float32)
        
        rgb = self.color_floats["g"] * np.ones((xyz.shape[0], xyz.shape[1]), dtype=np.float32) # (H, W)
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
    log = LogToF(
        tof_id=rospy.get_param("tof_id"),
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