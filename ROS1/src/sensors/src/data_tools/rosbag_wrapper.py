#!/usr/bin/env python

from rosbag import Bag
import numpy as np



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
    
    def write(
        self,
    ):
        pass 
        # with Bag("/home/spadmin/catkin_ws_ngp/data/DataSync/test.bag", 'w') as o: 
        #     o.write(b_topic, b_msg, b_time)
        

def main():

    rosbag = RosbagWrapper(
        bag_path="/home/spadmin/catkin_ws_ngp/data/DataSync/test.bag",
    )
    msgs, times = rosbag.read(
        topic="/USS1",
    )
    
    print(f"msgs: {msgs}")

if __name__ == "__main__":
    main()