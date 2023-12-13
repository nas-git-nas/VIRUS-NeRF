#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from sensors.msg import USS, TOF
import numpy as np
import struct

from pcl_tools.pcl_creator import PCLCreatorUSS, PCLCreatorTOF, PCLCreatorRS
from pcl_tools.pcl_coordinator import PCLCoordinator


class PCLPublisher():
    def __init__(
        self,
        sensor_id:str,
        pub_frame_id:str,
    ):
        self.sensor_id = sensor_id
        self.pub_frame_id = pub_frame_id
        
        # ROS
        self.sub = None
        self.pub = rospy.Publisher(sensor_id+"_pcl", PointCloud2, queue_size=10)
        
        # colors
        self.color_floats = {
            "w": struct.unpack('!f', bytes.fromhex('00FFFFFF'))[0],
            "r": struct.unpack('!f', bytes.fromhex('00FF0000'))[0],
            "g": struct.unpack('!f', bytes.fromhex('0000FF00'))[0],
            "b": struct.unpack('!f', bytes.fromhex('000000FF'))[0],
        }
        
        # sensor type specific variables
        if "USS" in self.sensor_id:
            self.pcl_creator = PCLCreatorUSS()
            self.color = self.color_floats["b"]
        elif "TOF" in self.sensor_id:
            self.pcl_creator = PCLCreatorTOF()
            self.color = self.color_floats["w"]
        elif "CAM" in self.sensor_id:
            self.pcl_creator = PCLCreatorRS()
            self.color = self.color_floats["g"]
        else:
            rospy.logerr(f"PCLPublisher.__init__: Unknown sensor_id: {self.sensor_id}")
        
        # sensor number specific variables
        if "1" in self.sensor_id:
            self.sub_frame_id = "CAM1"
        elif "3" in self.sensor_id:
            self.sub_frame_id = "CAM3"
        else:
            rospy.logerr(f"PCLPublisher.__init__: Unknown sensor_id: {self.sensor_id}")
            
        if self.sub_frame_id != self.pub_frame_id:
            self.pcl_coordinator = PCLCoordinator(
                source=self.sub_frame_id,
                target=self.pub_frame_id,
            )
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        if "USS" in self.sensor_id:
            msg_type = USS
        elif "TOF" in self.sensor_id:
            msg_type = TOF
        elif "CAM" in self.sensor_id:
            msg_type = Image
        else:
            rospy.logerr(f"PCLPublisher.subscribe: Unknown sensor_id: {self.sensor_id}")
            
        self.subscribe_uss = rospy.Subscriber("/"+self.sensor_id, msg_type, self._callback)       
        rospy.loginfo(f"PCLPublisher.subscribe: Subscribed to: {self.sensor_id}")
        
        rospy.spin()
        
    def _callback(
        self,
        msg,
    ):
        """
        Callback function for subscriber.
        """
        if "USS" in self.sensor_id or "TOF" in self.sensor_id:
            meas = msg.meas
        elif "CAM" in self.sensor_id:
            meas = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        else:
            rospy.logerr(f"PCLPublisher._callback: Unknown sensor_id: {self.sensor_id}")
            
        xyz = self.pcl_creator.meas2pc(
            meas=meas,
        )
        
        if self.sub_frame_id != self.pub_frame_id:
            xyz = self.pcl_coordinator.transformCoordinateSystem(
                xyz=xyz,
            )
            
        self._publishPCL(
            xyz=xyz,
        )
        
    def _publishPCL(
        self,
        xyz:np.array,
    ):
        """
        Publish pointcloud as Pointcloud2 ROS message.
        Args:
            xyz: pointcloud; np.array (N,3)
        """
        xyz = xyz[(xyz[:,0]!=np.NAN) & (xyz[:,1]!=np.NAN) & (xyz[:,2]!=np.NAN)]
        xyz = xyz.astype(dtype=np.float32)
        
        rgb = self.color * np.ones((xyz.shape[0], 1), dtype=np.float32) # (H, W)
        xyzrgb = np.concatenate((xyz, rgb), axis=1)
        
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = Header()
        pointcloud_msg.header.frame_id = self.pub_frame_id

        # Define the point fields (attributes)        
        pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        pointcloud_msg.height = 1
        pointcloud_msg.width = xyz.shape[0]

        # Float occupies 4 bytes. Each point then carries 12 bytes.
        pointcloud_msg.point_step = len(pointcloud_msg.fields) * 4 
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
        pointcloud_msg.is_bigendian = False # assumption
        pointcloud_msg.is_dense = True
        
        pointcloud_msg.data = xyzrgb.tobytes()
        self.pub_pointcloud.publish(pointcloud_msg)
        

def main():
    pub_pcl = PCLPublisher(
        sensor_id=rospy.get_param("sensor_id"),
        pub_frame_id=rospy.get_param("pub_frame_id"),
    )
    pub_pcl.subscribe()

if __name__ == '__main__':
    main()