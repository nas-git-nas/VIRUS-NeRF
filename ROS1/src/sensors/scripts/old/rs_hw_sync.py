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
import json
    
    
class HwSync():
    def __init__(
        self,
    ):
        """
        Verify hardware synchronization of realsense cameras
        """
        self.timestamp1 = 1.0
        self.timestamp2 = 1.0
        self.normalization = None
        
        # ROS
        self.topic1 = "/CAM1/color/metadata"
        self.topic2 = "/CAM3/color/metadata"
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
        metadata = json.loads(data.json_data)
        self.timestamp1 = metadata['frame_timestamp']
        # rospy.loginfo(f"LogRealSense._cbMeta1: hw timestamp: {metadata['frame_timestamp']}")
        
    def _cbMeta2(
        self,
        data:Metadata,
    ):
        """
        Callback for topic.
        Args:
            data: metadata from RealSense; Metadata
        """
        metadata = json.loads(data.json_data)
        self.timestamp2 = metadata['frame_timestamp']
        
        if self.normalization == None:
            self.normalization = self.timestamp2/self.timestamp1
        
        rospy.loginfo(f"LogRealSense._cbMeta2: timestamp ratio: {self.normalization*self.timestamp1/self.timestamp2}")

def main():
    hw_sync = HwSync()
    hw_sync.subscribe()

if __name__ == '__main__':
    main()