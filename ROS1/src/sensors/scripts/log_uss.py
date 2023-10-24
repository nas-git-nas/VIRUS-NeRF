#!/usr/bin/env python
import rospy
from std_msgs.msg import UInt32
from sensors.msg import uss

import cv2 as cv
import numpy as np
import pandas as pd
import os
import shutil
import time
    
    
class LogUSS():
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
        
        # delete-data of last measurement
        data_path = os.path.join(self.data_path, "data.csv")
        if os.path.isfile(data_path):
            os.remove(data_path)
        
        # data
        self.sequence = []
        self.seconds = []
        self.nano_seconds = []
        self.measurements = []
        
        # ROS
        rospy.init_node('log_uss', anonymous=True)
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        rospy.loginfo(f"LogRealSense.subscribe: Subscribe to: {self.topic}")
        rospy.Subscriber(self.topic, UInt32, self._callback)
        
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
            os.path.join(self.data_path, "data.csv")
        )
        rospy.loginfo(f"LogUSS.save: Save to: {self.data_path}")
        
    def _callback(
        self,
        data:UInt32,
    ):
        """
        Callback for topic.
        Args:
            data: data from USS; UInt32
        """
        if self.print_elapse_time:
            start = time.time()
        
        # log data
        self.sequence.append(data.header.seq)
        self.seconds.append(data.header.stamp.secs)
        self.nano_seconds.append(data.header.stamp.nsecs)
        self.measurements.append(data.data)
        
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


def main():
    log = LogUSS(
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