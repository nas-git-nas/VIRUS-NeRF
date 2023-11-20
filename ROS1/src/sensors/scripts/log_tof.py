#!/usr/bin/env python
import rospy
from std_msgs.msg import UInt32
from sensors.msg import uss
from sensors.msg import TOF
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import cv2 as cv
import numpy as np
import pandas as pd
import os
import shutil
import time
    
    
class LogToF():
    def __init__(
        self,
        topic:list,
        data_path:str,
        print_elapse_time:bool=False,
        publish_pointcloud:bool=False,
    ):
        """
        Log data from RealSense Node
        Args:
            topic: topic name, str
            data_path: path to save files; str
            print_elapse_time: wheter to print ellapse time of callback; bool
            publish_pointcloud: whether to publish pointcloud of tof measurements; bool
        """
        self.topic = topic
        self.data_path = data_path
        self.print_elapse_time = print_elapse_time
        self.publish_pointcloud = publish_pointcloud
        
        # delete-data of last measurement
        data_path = os.path.join(self.data_path, "data.csv")
        if os.path.isfile(data_path):
            os.remove(data_path)
        
        # data
        self.sequence = []
        self.seconds = []
        self.nano_seconds = []
        self.meas = []
        self.stds = []
        
        # ROS
        rospy.init_node('log_tof', anonymous=True)
        
        if self.publish_pointcloud:
            fov = 45 # field of view in degrees
            num_cells_in_row = 8
            
            cell_fov = np.deg2rad(fov) / num_cells_in_row
            angle_max = cell_fov * (num_cells_in_row - 1) / 2
            angle_min = - angle_max
            angles = np.linspace(angle_max, angle_min, num_cells_in_row)
            self.angles_y, self.angles_z = np.meshgrid(angles, angles, indexing="xy") # (N, N), (N, N)

            self.pub_pointcloud = rospy.Publisher('tof_pointcloud', PointCloud2, queue_size=10)
            
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        rospy.loginfo(f"LogRealSense.subscribe: Subscribe to: {self.topic}")
        rospy.Subscriber(self.topic, TOF, self._callback)
        
        rospy.spin()
        
    def save(
        self,
    ):
        """
        Save data.
        """
        # convert measurements and standard deviations
        meas = self._convertMeasurments(
            meas=self.meas,
        )
        stds = self._convertMeasurments(
            meas=self.stds,
        )
        
        # create pandas dataframe with data
        data = {
            "sequence": self.sequence,
            "time": np.array(self.seconds) + 1e-9*np.array(self.nano_seconds),
        }
        for i in range(meas.shape[1]):
            data["meas"+str(i)] = meas[:,i]
            data["stds"+str(i)] = stds[:,i]
        df = pd.DataFrame(
            data=data,
        )
       
        # save data
        df.to_csv(
            os.path.join(self.data_path, "data.csv")
        )
        rospy.loginfo(f"LogToF.save: Save to: {self.data_path}")
        
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
            
        # convert measurement
        meas = self._convertMeasurments(
            meas=data.meas,
        )
        stds = self._convertMeasurments(
            meas=data.stds,
        )
        
        # log data
        self.sequence.append(data.header.seq)
        self.seconds.append(data.header.stamp.secs)
        self.nano_seconds.append(data.header.stamp.nsecs)
        self.meas.append(meas)
        self.stds.append(stds)
        
        if self.publish_pointcloud:
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
        Measurements equal to 0 are invalid.
        Args:
            meas: measurements to convert; tuble (64)
        Returns:
            meas: measurments in meters, np.array of floats (8, 8)
        """
        meas = np.array(meas, dtype=float).reshape(8, 8)
        meas = meas[:,::-1]
        meas = meas.T
        
        meas[meas == 0] = np.nan
        return meas / 1000
    
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
        
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = Header()
        pointcloud_msg.header.frame_id = "ToF"

        # Define the point fields (attributes)        
        pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        pointcloud_msg.height = xyz.shape[0]
        pointcloud_msg.width = xyz.shape[1]

        # Float occupies 4 bytes. Each point then carries 12 bytes.
        pointcloud_msg.point_step = len(pointcloud_msg.fields) * 4 
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
        pointcloud_msg.is_bigendian = False # assumption
        pointcloud_msg.is_dense = True
        
        pointcloud_msg.data = xyz.tobytes()
        
        self.pub_pointcloud.publish(pointcloud_msg)


def main():
    log = LogToF(
        topic=rospy.get_param("topic"),
        data_path=rospy.get_param("path"),
        print_elapse_time=rospy.get_param("print_elapse_time"),
        publish_pointcloud=rospy.get_param("publish_pointcloud"),
    )
    try:
        log.subscribe()
    finally:
        log.save()

if __name__ == '__main__':
    main()