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
            
            # a = np.deg2rad(180)
            # self.xyz_rotation = np.array([[1.0, 0.0, 0.0],
            #                               [0.0, np.cos(a), -np.sin(a)],
            #                               [0.0, np.sin(a), np.cos(a)]])
            # self.xyz_translation = np.array([0.4, -0.2, 0.2])            

            if camera_id == "CAM1":
                fx, fy, cx, cy = (385.64675017644703, 385.11955977870474, 326.37231134404396, 243.8456231838371)
                self.rot_cam2sens = np.identity(3)
                self.trans_cam2sens = np.zeros((3))
                # self.rot_cam2sens = np.array([[0.9301244451630732, 0.01797626033041122, -0.3668042673887754],
                #                 [0.03279193955601337, 0.9907462305640599, 0.13170647411292763],
                #                 [0.36577753513609573, -0.1345316345237721, 0.9209278115586054]])
                # a = np.deg2rad(-30)
                # self.rot_cam2sens = np.array([[np.cos(a), 0.0, np.sin(a)],
                #                               [0.0, 1.0, 0.0],
                #                               [-np.sin(a), 0.0, np.cos(a)]])
                # self.trans_cam2sens = np.array([-0.43059516122700314, -0.17551265817557457, 0.009690706409878532])
               
            # elif camera_id == "CAM2":
            #     fx, fy, cx, cy = (381.8818263840882, 381.81960206089127, 318.76908626308136, 252.65292684183083)
            elif camera_id == "CAM3":
                fx, fy, cx, cy = (385.73700476936654, 385.36011039694336, 331.8394638201339, 242.75739627661798)
                # self.rot_cam2sens = np.identity(3)
                # self.trans_cam2sens = np.zeros((3))
                self.rot_cam2sens = np.array([[0.9301244451630732, 0.01797626033041122, -0.3668042673887754],
                                [0.03279193955601337, 0.9907462305640599, 0.13170647411292763],
                                [0.36577753513609573, -0.1345316345237721, 0.9209278115586054]])
                self.trans_cam2sens = np.array([-0.43059516122700314, -0.17551265817557457, 0.009690706409878532])
                
            self.directions = self._calcDirections(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                W=640,
                H=480,
            )
            
            self.T_cam1_lidar = np.array([[0.24761, -0.10048, 0.96364, 0.36199],
                                          [0.96679, -0.03926, -0.25252, -0.15161],
                                          [0.06321, 0.99416, 0.08742, -0.15014],
                                          [0.0, 0.0, 0.0, 1.0]])
            R_inv_temp = np.linalg.inv(self.T_cam1_lidar[:3,:3])
            t_inv_temp = - R_inv_temp @ self.T_cam1_lidar[:3,3]
            self.T_lidar_cam1 = np.zeros_like(self.T_cam1_lidar)
            self.T_lidar_cam1[:3,:3] = R_inv_temp
            self.T_lidar_cam1[:3,3] = t_inv_temp
            self.T_lidar_cam1[3,3] = 1
            
            a = np.deg2rad(50)
            b = np.deg2rad(-2)
            R = np.array([[np.cos(b), np.sin(b), 0.0, 0.0],
                          [-np.sin(b), np.cos(b), 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])
            R = np.array([[np.cos(a), 0.0, np.sin(a), -0.5],
                            [0.0, 1.0, 0.0, 0.0],
                            [-np.sin(a), 0.0, np.cos(a), -0.2],
                            [0.0, 0.0, 0.0, 1.0]]) @ R
            R = np.array([[0.911262, -0.234436, 0.338589, 0.097039],
                          [0.265099, 0.963094, -0.046639, 0.088046],
                          [-0.315159, 0.132260, 0.939779, 0.337549],
                          [0.0, 0.0, 0.0, 1.0]]) @ R
            R = np.array([[0.998423, 0.056128, -0.000969, 0.060750],
                          [-0.056028, 0.997414, 0.045011, -0.046822],
                          [0.003493, -0.044886, 0.998986, -0.035082],
                          [0.0, 0.0, 0.0, 1.0]]) @ R
            R = np.array([[0.974062, 0.054679, 0.219576, -0.581597],
                          [-0.072013, 0.994821, 0.071725, -0.095861],
                          [-0.214517, -0.085677, 0.972955, 0.383083],
                          [0.0, 0.0, 0.0, 1.0]]) @ R

            self.T_cam3_cam1 = R
            
        
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
        
        # if self.publish_pointcloud and isinstance(self.directions, np.ndarray):
        if self.publish_pointcloud:
            # rospy.loginfo(f"LogRealSense._callback: pup pointcloud ...")
            depth = self._convertDepth(
                depth=img,
            )
            xyz = self._depth2pointcloud(
                depth=depth,
            )
            # xyz = self._transformPointcloud(
            #     xyz=xyz,
            # )
            # self._publishPointcloud(
            #     xyz=xyz[::1,::1],
            #     mask=(depth<6.0)[::1,::1],
            #     header=data.header
            # )
            self._publishPointcloud(
                xyz=xyz,
                mask=(depth<6.0),
                header=data.header
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
        
        # if self.publish_pointcloud:
        #     self.directions = self._calcDirections(
        #         fx=self.camera_info["fx"],
        #         fy=self.camera_info["fy"],
        #         cx=self.camera_info["cx"],
        #         cy=self.camera_info["cy"],
        #         W=self.camera_info["W"],
        #         H=self.camera_info["H"],
        #     )
        
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
        # directions = np.stack((dir_z, dir_x, dir_y), axis=2)
        directions = np.stack((dir_x, dir_y, dir_z), axis=2)
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
        return depth[:,:,None] * self.directions
    
    def _transformPointcloud(
        self,
        xyz:np.array,
    ):
        """
        Transform pointcloud
        Args:
            xyz: pointcloud; np.array (H, W, 3)
        """
        # H = xyz.shape[0]
        # W = xyz.shape[1]
        # xyz = xyz.reshape(H*W, 3).T, # (3, H*W)
        
        # xyz = self.rot_img2cam @ xyz
        # xyz = (self.rot_cam2sens @ xyz) + self.trans_cam2sens[:, None] # (3, H*W)
        # xyz = self.rot_sens2robot @ xyz # (3, H*W)
        
        # xyz = xyz.T.reshape(H, W, 3) # (H, W, 3)
        
        H = xyz.shape[0]
        W = xyz.shape[1]
        xyz = xyz.reshape(H*W, 3).T # (3, H*W)
        xyz = np.concatenate((xyz, np.ones((1, H*W))), axis=0) # (4, H*W)
        
        if self.camera_id == "CAM3":
            xyz = self.T_cam3_cam1 @ xyz
        
        # xyz = self.T_cam1_lidar @ xyz
        
        xyz = xyz[:3,:] # (3, H*W)
        xyz = xyz.T.reshape(H, W, 3)
        
        return xyz
        
    def _publishPointcloud(
        self,
        xyz:np.array,
        mask:np.array,
        header:Header,
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
            
        xyz = xyz.reshape(-1 ,3) # (H*W, 3)
        mask = mask.flatten()
        
        # xyz = xyz[mask]
        # xyz = xyz[xyz[:,1]>-1.5]
        
            
        if self.camera_id == "CAM1":
            color = self.color_floats["w"]
        elif self.camera_id == "CAM3":
            color = self.color_floats["b"]
        
        rgb = color * np.ones((xyz.shape[0]), dtype=np.float32) # (H, W)
        xyzrgb = np.concatenate((xyz, rgb[:, None]), axis=1)
        # rgb = self.color_floats["w"] * np.ones_like(mask, dtype=np.float32) # (H, W)
        # rgb[mask] = self.color_floats["r"]
        # xyzrgb = np.concatenate((xyz, rgb[:,:,None]), axis=2)
        
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = Header()
        # pointcloud_msg.header.frame_id = "rslidar"
        pointcloud_msg.header.frame_id = "CAM1" if self.camera_id=="CAM1" else "CAM3"
        pointcloud_msg.header.stamp.secs = header.stamp.secs
        pointcloud_msg.header.stamp.nsecs = header.stamp.nsecs

        # Define the point fields (attributes)        
        pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        pointcloud_msg.height = 1 #xyz.shape[1]
        pointcloud_msg.width = xyz.shape[0]

        # Float occupies 4 bytes. Each point then carries 12 bytes.
        pointcloud_msg.point_step = len(pointcloud_msg.fields) * 4 
        # pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width * pointcloud_msg.height
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
        pointcloud_msg.is_bigendian = False # assumption
        pointcloud_msg.is_dense = True
        
        pointcloud_msg.data = xyzrgb.tobytes()
        
        self.pub_pointcloud.publish(pointcloud_msg)
        
        # rospy.loginfo(f"LogRealSense._publishPointcloud: published")
        
        

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