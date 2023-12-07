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


class FilterLidar():
    def __init__(
        self,
    ) -> None:
        
        # ROS
        self.topic = "/LIDAR/rslidar_points"
        self.subscribe = None
        self.pub_pointcloud = rospy.Publisher('rslidar_points_filtered', PointCloud2, queue_size=10)
        rospy.init_node('filter_lidar', anonymous=True)
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        self.subscribe_depth = rospy.Subscriber(self.topic, PointCloud2, self._filter)
        rospy.loginfo(f"FilterLidar.subscribe: Subscribed to: {self.topic}")
        rospy.spin()
        
    def _filter(
        data:PointCloud2,
    ):
        pass