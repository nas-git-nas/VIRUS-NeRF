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
    
    
class LogRealSense():
    def __init__(
        self,
        camera_id:list,
        data_dir:str,
        print_elapse_time:bool=False,
        publish_pointcloud:bool=False,
    ):
        """
        Log data from RealSense Node
        Args:
            camera_id: either CAM1, CAM2 or CAM3; str
            data_dir: path where to save data; str
            print_elapse_time: whether to print ellapse time of callback; bool
            publish_pointcloud: whether to publish pointcloud of depth measurements; bool
        """
        self.camera_id = camera_id
        self.camera_dir = os.path.join(data_dir, camera_id)
        self.print_elapse_time = print_elapse_time
        self.publish_pointcloud = publish_pointcloud
        
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
        rospy.init_node('rs_log', anonymous=True)
        
        # pointcloud
        if self.publish_pointcloud:
            self.camera_info = None
            self.directions = None
            self.rgb = None
            self.pub_pointcloud = rospy.Publisher('rgbd_pointcloud', PointCloud2, queue_size=10)
            
            self.color_floats = {
                "w": struct.unpack('!f', bytes.fromhex('00FFFFFF'))[0],
                "r": struct.unpack('!f', bytes.fromhex('00FF0000'))[0],
                "g": struct.unpack('!f', bytes.fromhex('0000FF00'))[0],
                "b": struct.unpack('!f', bytes.fromhex('000000FF'))[0],
            }
        
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
        
        if self.publish_pointcloud and isinstance(self.directions, np.ndarray):
            # rospy.loginfo(f"LogRealSense._callback: pup pointcloud ...")
            depth = self._convertDepth(
                depth=img,
            )
            xyz = self._depth2pointcloud(
                depth=depth,
            )
            self._publishPointcloud(
                xyz=xyz,
                mask=np.ones((xyz.shape[0], xyz.shape[1]), dtype=bool),
            )
        
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
        
        if self.publish_pointcloud:
            self.directions = self._calcDirections(
                fx=self.camera_info["fx"],
                fy=self.camera_info["fy"],
                cx=self.camera_info["cx"],
                cy=self.camera_info["cy"],
                W=self.camera_info["W"],
                H=self.camera_info["H"],
            )
        
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
        
    def _calcDirections(
        self,
        fx:float,
        fy:float,
        cx:float,
        cy:float,
        W:int,
        H:int,
    ):
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
        return directions / np.linalg.norm(directions, axis=2, keepdims=True)
    
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
        depth[depth>1.0] = 0
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
        
        rgb = self.color_floats["w"] * np.ones((xyz.shape[0], xyz.shape[1]), dtype=np.float32) # (H, W)
        xyzrgb = np.concatenate((xyz, rgb[:,:,None]), axis=2)
        # rgb = self.color_floats["w"] * np.ones_like(mask, dtype=np.float32) # (H, W)
        # rgb[mask] = self.color_floats["r"]
        # xyzrgb = np.concatenate((xyz, rgb[:,:,None]), axis=2)
        
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
        # pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width * pointcloud_msg.height
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
        pointcloud_msg.is_bigendian = False # assumption
        pointcloud_msg.is_dense = True
        
        pointcloud_msg.data = xyzrgb.tobytes()
        
        self.pub_pointcloud.publish(pointcloud_msg)
            
        

def main():
    log = LogRealSense(
        camera_id=rospy.get_param("camera_id"),
        data_dir=rospy.get_param("data_dir"),
        print_elapse_time=rospy.get_param("print_elapse_time"),
        publish_pointcloud=rospy.get_param("publish_pointcloud"),
    )
    try:
        log.subscribe()
    finally:
        log.save()

if __name__ == '__main__':
    main()