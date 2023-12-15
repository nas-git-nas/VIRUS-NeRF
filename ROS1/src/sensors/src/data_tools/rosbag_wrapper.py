#!/usr/bin/env python

from rosbag import Bag
import rospy
import numpy as np
from fnmatch import fnmatchcase
import copy
import os



class RosbagWrapper():
    def __init__(
        self,
        bag_path:str,
    ) -> None:
        self.bag_path = bag_path
        
    def read(
        self,
        topic:str,
    ):
        """
        Get messages from a particular topic.
        Args:
            topic: topic name; str
        Returns:
            meass: measurements; np.array of floats (N)
            times: times of measurements in seconds; np.array of floats (N)
        """
        with Bag(self.bag_path, 'r') as bag:
            
            meass = []
            times = []
            for b_topic, b_msg, b_time in bag:
                
                if b_topic != topic:
                    continue
                
                if "USS" in topic:
                    meas, time = self._readMsgUSS(
                        msg=b_msg,
                    )
                elif "TOF" in topic:
                    meas, time = self._readMsgToF(
                        msg=b_msg,
                    )
                elif "CAM" in topic:
                    meas = None
                    time = self._readMsgRS(
                        msg=b_msg,
                    )
                meass.append(meas)
                times.append(time)
                
        return np.array(meass), np.array(times)
    
    def writeTimeStamp(
        self,
        bag_path_sync:str,
        topic_async:str,
        topic_sync:str,
        time_async:np.array,
        time_sync:np.array,
    ):
        """
        Write messages to a particular topic.
        Args:
            bag_path_sync: path to bag file to write to; str
            topic_async: topic name to copy data from; str
            topic_sync: topic name to write data to; str
            time_async: time of msg to copy data from; np.array of floats (N)
            time_sync: synchronized time; np.array of floats (N)
        """
        print(f"topic_async={topic_async}")
        
        with Bag(bag_path_sync, 'w') as bag:
            
            counter = 0
            for b_topic, b_msg, b_time in Bag(self.bag_path).read_messages():
                
                if b_topic != topic_async:
                    continue
                
                # enter while loop and save message if this source message corresponds to a target message
                # enter multiple times the while loop if this source message corresponds to multiple target messages
                b_msg_time = b_msg.header.stamp.to_sec()
                while b_msg_time == time_async[counter]:
                    ros_time = rospy.Time.from_sec(time_sync[counter])
                    b_msg.header.stamp = ros_time
                    bag.write(topic_sync, b_msg, ros_time)
                
                    counter += 1
                    if counter >= len(time_sync):
                        break
                
                if counter >= len(time_sync):
                    break
                
        if counter != len(time_sync):
            print(f"ERROR: RosbagWrapper.writeTimeStamp: counter={counter} != len(time_sync)={len(time_sync)}")
            
    def cropBag(
        self,
        bag_path_to_crop:str,
        msgs_range:tuple,
    ):
        """
        Write messages to a particular topic.
        Args:
            bag_path_to_crop: path to bag file that should be croped; str
            msgs_range: number of messages to copy; tuple of ints (start, end)
        """
        with Bag(self.bag_path, 'w') as bag:
            
            counter = 0
            for b_topic, b_msg, b_time in Bag(bag_path_to_crop).read_messages():
                
                counter += 1
                if counter < msgs_range[0]:
                    continue
                if counter >= msgs_range[1]:
                    break
                    
                bag.write(b_topic, b_msg, b_time)
                
        print(f"INFO: RosbagWrapper.cropBag: {counter-msgs_range[0]} messages written to {self.bag_path}")
                
    def merge(
        self,
        bag_path_out:str,
        bag_paths_ins:list,
        keep_topics:list=None,
        delete_ins:bool=False,
    ):
        """
        Merge multiple bag files into one.
        Args:
            bag_path_out: path to bag file to write to; str
            bag_paths_ins: list of paths to bag files to read from; list of str
            keep_topics: list of list that indicates for each bag file which topics to read,
                         if topic list is equal to 'all', all topics are merged of this list; list of list of str
            delete_ins: list that indicates to delete input bag; list of bool
        """
        if keep_topics is None:
            keep_topics = ["all"] * len(bag_paths_ins)
        
        bag_path_out_origianl = copy.copy(bag_path_out)
        if bag_path_out in bag_paths_ins:
            bag_path_out = bag_path_out.replace(".bag", "_this_is_a_random_string.bag")

        with Bag(bag_path_out, 'w') as o: 
            
            for i, ifile in enumerate(bag_paths_ins):
                with Bag(ifile, 'r') as ib:
                    
                    for topic, msg, t in ib:
                        if (keep_topics[i] == "all") or (topic in keep_topics[i]):
                            o.write(topic, msg, t)
                        
        if bag_path_out != bag_path_out_origianl:
            os.rename(bag_path_out, bag_path_out_origianl)
        
        
        for i, ifile in enumerate(bag_paths_ins):
            if delete_ins[i]:
                os.remove(ifile)


    def _readMsgUSS(
        self,
        msg:object,
    ):
        """
        Read USS message from rosbag.
        Args:
            msg: rosbag message; USS
        Returns:
            meas: USS measurement; float
            time: time of measurement in seconds; float
        """
        meas = msg.meas
        time = msg.header.stamp.to_sec()
        return meas, time
    
    def _readMsgToF(
        self,
        msg:object,
    ):
        """
        Read USS message from rosbag.
        Args:
            msg: rosbag message; ToF
        Returns:
            meas: ToF measurement; float
            time: time of measurement in seconds; float
        """
        meas = msg.meas
        time = msg.header.stamp.to_sec()
        return meas, time
    
    def _readMsgRS(
        self,
        msg:object,
    ):
        """
        Read USS message from rosbag.
        Args:
            msg: rosbag message; sensor_msgs/Image 
        Returns:
            time: time of measurement in seconds; float
        """
        time = msg.header.stamp.to_sec()
        return time
        

def main():

    bag_wrap = RosbagWrapper(
        bag_path="/home/spadmin/catkin_ws_ngp/data/DataSync/test.bag",
    )
    bag_wrap.cropBag(
        bag_path_to_crop="/home/spadmin/catkin_ws_ngp/data/DataSync/office_2.bag",
        msgs_range=(150740, 160741),
    )

if __name__ == "__main__":
    main()