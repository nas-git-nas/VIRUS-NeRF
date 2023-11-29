#!/usr/bin/env python
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
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
    
    
class LogRealSense():
    def __init__(
        self,
        camera_id:list,
        data_dir:str,
        print_elapse_time:bool=False,
    ):
        """
        Log data from RealSense Node
        Args:
            topic: topic name, str
            data_path: path to save files; str
            print_elapse_time: whether to print ellapse time of callback; bool
        """
        self.camera_id = camera_id
        self.camera_dir = os.path.join(data_dir, camera_id)
        self.print_elapse_time = print_elapse_time
        
        # delete last measurement
        if os.path.exists(self.camera_dir):
            shutil.rmtree(self.camera_dir)
        os.mkdir(os.path.join(self.camera_dir))
        os.mkdir(os.path.join(self.camera_dir, "depth"))
        os.mkdir(os.path.join(self.camera_dir, "depth", "imgs"))
        os.mkdir(os.path.join(self.camera_dir, "rgb"))
        os.mkdir(os.path.join(self.camera_dir, "rgb", "imgs"))
        
        # data conversion
        self.cv_bridge = CvBridge()
        
        # meta data
        self.meta_depth = {
            "seq": [],
            "s": [],
            "ns": [],
        }
        self.meta_rgb = {
            "seq": [],
            "s": [],
            "ns": [],
        }
        self.camera_info = {
            "fx": None,
            "fy": None,
            "cx": None,
            "cy": None,
            "W": None,
            "H": None,
        }
        self.timestamps = {
            "frame": [],
            "hw": [],
            "sensor": [],
        }
        
        # ROS
        self.topic_depth = "/" + self.camera_id + "/aligned_depth_to_color/image_raw"
        self.topic_rgb = "/" + self.camera_id + "/color/image_raw"
        self.topic_rgb_info = "/" + self.camera_id + "/color/camera_info"
        self.topic_rgb_meta = "/" + self.camera_id + "/color/metadata"
        self.subscribe_depth = None
        self.subscribe_rgb = None
        self.subscribe_rgb_info = None
        self.subscribe_rgb_meta = None
        rospy.init_node('log_realsense', anonymous=True)
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        self.subscribe_depth = rospy.Subscriber(self.topic_depth, Image, self._cbDepthImage)
        self.subscribe_rgb = rospy.Subscriber(self.topic_rgb, Image, self._cbRGBImage)
        self.subscribe_rgb_info = rospy.Subscriber(self.topic_rgb_info, CameraInfo, self._cbRGBInfo)
        self.subscribe_rgb_meta = rospy.Subscriber(self.topic_rgb_meta, Metadata, self._cbRGBMeta)
        
        rospy.loginfo(f"LogRealSense.subscribe: Subscribed to: {self.topic_depth}, {self.topic_rgb}, {self.topic_rgb_info} and {self.topic_rgb_meta}")
        rospy.spin()
        
    def save(
        self,
    ):
        """
        Save meta-data.
        """
        df_depth = pd.DataFrame(
            {
                "sequence": self.meta_depth["seq"],
                "time": np.array(self.meta_depth["s"]) + 1e-9*np.array(self.meta_depth["ns"])
            }
        )
        df_rgb = pd.DataFrame(
            {
                "sequence": self.meta_rgb["seq"],
                "time": np.array(self.meta_rgb["s"]) + 1e-9*np.array(self.meta_rgb["ns"])
            }
        )
        df_rgb_info = pd.DataFrame(
            self.camera_info,
            index=[0],
        )
        df_rgb_meta = pd.DataFrame(self.timestamps)
       
        df_depth.to_csv(
            os.path.join(self.camera_dir, "depth", "meta_data.csv"),
            index=False,
        )
        df_rgb.to_csv(
            os.path.join(self.camera_dir, "rgb", "meta_data.csv"),
            index=False,
        )
        df_rgb_info.to_csv(
            os.path.join(self.camera_dir, "camera_info.csv"),
            index=False,
        )
        df_rgb_meta.to_csv(
            os.path.join(self.camera_dir, "camera_meta.csv"),
            index=False,
        )
        rospy.loginfo(f"LogRealSense.save: Saved to: {self.camera_dir}")
            
    def _cbDepthImage(
        self,
        data:Image,
    ):
        """
        Callback for topic.
        Args:
            data: data from RealSense; Image
        """
        if self.print_elapse_time:
            start = time.time()
        
        # log meta data
        self.meta_depth["seq"].append(data.header.seq)
        self.meta_depth["s"].append(data.header.stamp.secs)
        self.meta_depth["ns"].append(data.header.stamp.nsecs)

        # convert and save img
        img = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        status = cv.imwrite(
            filename=os.path.join(self.camera_dir, "depth", "imgs", f"img{self.meta_depth['seq'][-1]}.png"), 
            img=img,
        )
        
        if not status:
            rospy.logwarn(f"LogRealSense._cbDepthImage: img save status: {status}")
        
        if self.print_elapse_time:
            rospy.loginfo(f"LogRealSense._cbDepthImage: elapse time: {(time.time()-start):.3f}s")

    def _cbRGBImage(
        self,
        data:Image,
    ):
        """
        Callback for topic.
        Args:
            data: data from RealSense; Image
        """
        if self.print_elapse_time:
            start = time.time()
        
        # log meta data
        self.meta_rgb["seq"].append(data.header.seq)
        self.meta_rgb["s"].append(data.header.stamp.secs)
        self.meta_rgb["ns"].append(data.header.stamp.nsecs)

        # convert and save img
        img = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        status = cv.imwrite(
            filename=os.path.join(self.camera_dir, "rgb", "imgs", f"img{self.meta_rgb['seq'][-1]}.png"), 
            img=img,
        )
        
        if not status:
            rospy.logwarn(f"LogRealSense._cbRGBImage: img save status: {status}")
        
        if self.print_elapse_time:
            rospy.loginfo(f"LogRealSense._cbImageRGB: elapse time: {(time.time()-start):.3f}s")
              
    def _cbRGBInfo(
        self,
        data:CameraInfo,
    ):
        """
        Callback function to get camera infos
        Args:
            data: intrinsic camera parameters; CameraInfo
        """   
        if self.print_elapse_time:
            start = time.time()
                 
        self.camera_info = {
            "fx": data.K[0],
            "fy": data.K[4],
            "cx": data.K[2],
            "cy": data.K[5],
            "W": data.width,
            "H": data.height,
        }
        self.subscribe_rgb_info.unregister()
        
        if self.print_elapse_time:
            rospy.loginfo(f"LogRealSense._cbRGBInfo: elapse time: {(time.time()-start):.3f}s")
            
    def _cbRGBMeta(
        self,
        data:Metadata,
    ):
        """
        Callback function to get timestamps
        Args:
            data: camera meta data; Metadata
        """
        metadata = json.loads(data.json_data)

        self.timestamps["frame"].append(metadata["frame_timestamp"])
        self.timestamps["hw"].append(metadata["hw_timestamp"])
        self.timestamps["sensor"].append(metadata["sensor_timestamp"])
        
        

def main():
    log = LogRealSense(
        camera_id=rospy.get_param("camera_id"),
        data_dir=rospy.get_param("data_dir"),
        print_elapse_time=rospy.get_param("print_elapse_time"),
    )
    try:
        log.subscribe()
    finally:
        log.save()

if __name__ == '__main__':
    main()