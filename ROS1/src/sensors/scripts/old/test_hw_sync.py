#!/usr/bin/env python
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from realsense2_camera.msg import Metadata
from cv_bridge import CvBridge
import pyrealsense2
import cv2 as cv
import numpy as np
import pandas as pd
import os
import shutil
import time
import struct
    
    
class HwSync():
    def __init__(
        self,
    ):
        """
        Verify hardware synchronization of realsense cameras
        """
        # ROS
        self.topic1 = "/CAM1/color/metadata"
        self.topic2 = "/CAM2/color/metadata"
        rospy.init_node('log_realsense', anonymous=True)
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        rospy.loginfo(f"LogRealSense.subscribe: Subscribe to: {self.topic1} and {self.topic2}")
        rospy.Subscriber(self.topic1, Metadata, self._cbMeta1)
        rospy.Subscriber(self.topic2, Metadata, self._cbMeta2)
        
        rospy.spin()
        
    def _cbMeta1(
        self,
        data:Metadata,
    ):
        """
        Callback for topic.
        Args:
            data: metadata from RealSense; Metadata
        """
        rospy.loginfo(f"LogRealSense._cbMeta1: hw timestamp: {data['frame_timestamp']}")
        
    def _cbMeta1(
        self,
        data:Metadata,
    ):
        """
        Callback for topic.
        Args:
            data: metadata from RealSense; Metadata
        """
        rospy.loginfo(f"LogRealSense._cbMeta2: hw timestamp: {data['frame_timestamp']}")
        

def main():
    hw_sync = HwSync()
    hw_sync.subscribe()

if __name__ == '__main__':
    main()