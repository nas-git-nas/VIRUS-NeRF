#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2 as cv
import numpy as np
import pandas as pd
import os
import shutil
import time
    
    
class LogRealSense():
    def __init__(
        self,
        topic:list,
        data_path:str,
        print_elapse_time:bool=False
    ):
        """
        Log data from RealSense Node
        Args:
            topic: topic name, str
            data_path: path to save files; str
            print_elapse_time: wheter to print ellapse time of callback; bool
        """
        self.topic = topic
        self.data_path = data_path
        self.print_elapse_time = print_elapse_time
        
        # delete images and meta-data of last measurement
        imgs_path = os.path.join(self.data_path, "imgs")
        if os.path.exists(imgs_path):
            shutil.rmtree(imgs_path)
        os.mkdir(imgs_path)
        
        metadata_path = os.path.join(self.data_path, "meta_data.csv")
        if os.path.isfile(metadata_path):
            os.remove(metadata_path)
        
        # data conversion
        self.cv_bridge = CvBridge()
        
        # meta data
        self.sequence = []
        self.seconds = []
        self.nano_seconds = []
        
        # ROS
        rospy.init_node('log_realsense', anonymous=True)
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        rospy.loginfo(f"LogRealSense.subscribe: Subscribe to: {self.topic}")
        rospy.Subscriber(self.topic, Image, self._callback)
        
        rospy.spin()
        
    def save(
        self,
    ):
        """
        Save meta-data.
        """
        df = pd.DataFrame(
            {
                "sequence": self.sequence,
                "time": np.array(self.seconds) + 1e-9*np.array(self.nano_seconds)
            }
        )
       
        df.to_csv(
            os.path.join(self.data_path, "meta_data.csv")
        )
        rospy.loginfo(f"LogRealSense.save: Save to: {self.data_path}")
        
    def _callback(
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
        self.sequence.append(data.header.seq)
        self.seconds.append(data.header.stamp.secs)
        self.nano_seconds.append(data.header.stamp.nsecs)

        # convert and save img
        img = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        status = cv.imwrite(
            filename=os.path.join(self.data_path, "imgs", f"img{self.sequence[-1]}.png"), 
            img=img,
        )
        
        if not status:
            rospy.logwarning(f"LogRealSense._callback: img save status: {status}")
        
        if self.print_elapse_time:
            rospy.loginfo(f"LogRealSense._callback: elapse time: {(time.time()-start):.3f}s")


def main():
    log = LogRealSense(
        topic=rospy.get_param("topic"),
        data_path=rospy.get_param("path"),
        print_elapse_time=rospy.get_param("print_elapse_time"),
    )
    try:
        log.subscribe()
    finally:
        log.save()

if __name__ == '__main__':
    main()