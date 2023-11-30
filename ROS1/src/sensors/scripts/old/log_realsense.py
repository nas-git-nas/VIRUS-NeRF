#!/usr/bin/env python
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import pyrealsense2
import cv2 as cv
import numpy as np
import pandas as pd
import os
import shutil
import time
import struct
    
    
class LogRealSense():
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
            print_elapse_time: whether to print ellapse time of callback; bool
            publish_pointcloud: whether to publish pointcloud of depth measurements; bool
        """
        self.topic = topic
        self.data_path = data_path
        self.print_elapse_time = print_elapse_time
        self.publish_pointcloud = publish_pointcloud
        
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
        
        if self.publish_pointcloud:
            self.camera_info = None
            self.directions = None
            self.rgb = None
            self.pub_pointcloud = rospy.Publisher('rgbd_pointcloud', PointCloud2, queue_size=10)
            
            self.color_floats = {
                "w": struct.unpack('!f', bytes.fromhex('00FFFFFF'))[0],
                "r": struct.unpack('!f', bytes.fromhex('00FF0000'))[0],
            }
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        rospy.loginfo(f"LogRealSense.subscribe: Subscribe to: {self.topic}")
        rospy.Subscriber(self.topic, Image, self._cbImage)
        rospy.Subscriber("/camera/color/image_raw", Image, self._cbImageColor)
        
        if self.publish_pointcloud:
            rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self._cbCameraInfo)
        
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
        
    def _cbImage(
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
        img = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        status = cv.imwrite(
            filename=os.path.join(self.data_path, "imgs", f"img{self.sequence[-1]}.png"), 
            img=img,
        )
        # print(f"LogRealSense._callback: img dtype: {img.dtype}, shape: {img.shape}, np: {isinstance(img, np.ndarray)}")
        
        if not status:
            rospy.logwarn(f"LogRealSense._callback: img save status: {status}")
            
        if self.publish_pointcloud and isinstance(self.directions, np.ndarray):
            # rospy.loginfo(f"LogRealSense._callback: pup pointcloud ...")
            depth = self._convertDepth(
                depth=img,
            )
            mask = self._identifyCorners(
                depth=depth,
            )
            xyz = self._depth2pointcloud(
                depth=depth,
            )
            self._publishPointcloud(
                xyz=xyz,
                mask=mask,
            )
        
        if self.print_elapse_time:
            rospy.loginfo(f"LogRealSense._callback: elapse time: {(time.time()-start):.3f}s")

    def _cbImageColor(
        self,
        data:Image,
    ):
        """
        Callback for topic.
        Args:
            data: data from RealSense; Image
        """
        # convert and save img
        rgb = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        # print(f"LogRealSense._cbImageColor: img dtype: {rgb.dtype}, "
        #       +f"shape: {rgb.shape}, np: {isinstance(rgb, np.ndarray)}")
        rgb = self._convertColor(
            rgb=rgb,
        )
        self.rgb = rgb
              
    def _cbCameraInfo(
        self,
        data:CameraInfo,
    ):
        """
        Callback function to get camera infos
        Assuming that camera is upside down. This means the pixel 
            [0,0]       = bottom right corner,
            [H-1,0]     = top right corner,
            [0,W-1]     = bottom left corner,
            [h-1,W-1]   = top left corner.
        Therefore, aus and vs are from 0 to W/H and not the inverse.
        Args:
            data: intrinsic camera parameters; CameraInfo
        """
        fx = data.K[0]
        fy = data.K[4]
        cx = data.K[2]
        cy = data.K[5]
        W = data.width
        H = data.height
        
        # coordinate system: intrinsic
        #   dir_x <-> width
        #   dir_y <-> height
        #   dir_z <-> viewing direction
        us, vs = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        dir_x = (us - cx + 0.5) / fx
        dir_y = (vs - cy + 0.5) / fy
        dir_z = np.ones_like(us)
        
        # coordinate system: ROS
        #   dir_x <-> viewing direction
        #   dir_y <-> width
        #   dir_z <-> height
        directions = np.stack((dir_z, dir_x, dir_y), axis=2)
        self.directions = directions / np.linalg.norm(directions, axis=2, keepdims=True)
        
    def _convertDepth(
        self,
        depth:np.array,
    ):
        """
        Convert depth measurement.
        Args:
            depth: depth measurement; np.array of uint16 (H, W)
        Returns:
            depth: converted dpeths; np.array of float32 (H, W)
        """
        depth = 0.001 * depth.astype(dtype=np.float32)
        return depth # flip depth
    
    def _convertColor(
        self,
        rgb:np.array,
    ):
        """
        Convert color image when camera is upside down. This means the input pixel 
            [0,0]       = bottom right corner,
            [H-1,0]     = top right corner,
            [0,W-1]     = bottom left corner,
            [h-1,W-1]   = top left corner.
        Args:
            rgb: color image; np.array of uint8 (H, W, 3)
        Returns:
            rgb: converted color image; np.array of uint8 (H, W, 3)
        """
        return rgb[::-1,::-1]
    
    def _identifyCorners(
        self,
        depth:np.array,
    ):
        """
        Identify corners of calibration target. Target should be squared,
        brighter as the background and closer to the camera than any other object.
        Args:
            depth: depth image; np.array of float32 (H, W)
        Returns:
            mask: mask indicating targets; np.array of bool (H, W)
        """
        mask = np.zeros_like(depth, dtype=bool)
        
        if isinstance(self.rgb, np.ndarray):
            threshold_dist = 1.0
            mask[depth < threshold_dist] = True
            
            # threshold_color = 100
            # mask[np.sum(self.rgb>threshold_color, axis=2) > 2] = True
        
        return mask
        

    def _depth2pointcloud(
        self,
        depth:np.array,
    ):
        """
        Convert depth measurements to pointcloud.
        Assuming x points into the viewing direction and z upwards.
        Define dim1=z-axis and dim2=y-axis going from larger to smaller values.
        Args:
            depth: depth image; np.array of float32 (H, W)
        Returns:
            xyz: pointcloud; np.array (H, W, 3)
        """
        return depth[:,:,None] * self.directions
        
        
    def _publishPointcloud(
        self,
        xyz:np.array,
        mask:np.array,
    ):
        """
        Publish pointcloud as Pointcloud2 ROS message.
        Args:
            xyz: pointcloud; np.array (H, W, 3)
            mask: mask indicating targets; np.array of bool (H, W)
        """
        xyz[xyz==np.NAN] = 0
        if xyz.dtype != np.float32:
            xyz = xyz.astype(dtype=np.float32)
            
        rgb = self.color_floats["w"] * np.ones_like(mask, dtype=np.float32) # (H, W)
        rgb[mask] = self.color_floats["r"]
        xyzrgb = np.concatenate((xyz, rgb[:,:,None]), axis=2)
        
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = Header()
        pointcloud_msg.header.frame_id = "RGBD"

        # Define the point fields (attributes)        
        pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        pointcloud_msg.height = xyz.shape[1]
        pointcloud_msg.width = xyz.shape[0]

        # Float occupies 4 bytes. Each point then carries 12 bytes.
        pointcloud_msg.point_step = len(pointcloud_msg.fields) * 4 
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width * pointcloud_msg.height
        pointcloud_msg.is_bigendian = False # assumption
        pointcloud_msg.is_dense = True
        
        pointcloud_msg.data = xyzrgb.tobytes()
        
        self.pub_pointcloud.publish(pointcloud_msg)


def main():
    log = LogRealSense(
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