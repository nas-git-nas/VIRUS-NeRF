#!/usr/bin/env python
import numpy as np
from abc import ABC, abstractmethod


class PCLCreator():
    
    def __init__(
        self,
    ):
        pass
    
    @abstractmethod 
    def meas2depth(
        self,
        meas:np.array,
    ):
        pass
    
    def meas2pc(
        self,
        meas:np.array,
    ):
        """
        Convert depth measurments to pointcloud depending on sensor type.
        Args:
            depth: depth measurments; numpy array of shape (N,)
        Returns:
            xyz: pointcloud; numpy array of shape (N,3)
        """
        depth = self.meas2depth(
            meas=meas,
        )
        xyz = self.depth2pc(
            depth=depth,
        )
        return xyz
    
    def depth2pc(
        self,
        depth:np.array,
    ):
        """
        Converting depth into 3D point cloud.
        Args:
            depth: converted depth measurement; np.array (H, W)
        Returns:
            xyz: point cloud; np.array (H*W, 3)
        """
        depth = depth.reshape(-1, 1) # (H*W, 1)
        xyz = self.directions * depth # (H*W, 3)
        return xyz
    
    def fovDirections(
        self,
        fov_xy:list,
        W:int,
        H:int,
    ):
        """
        Calculate directions given a field of view.
        Coordinate system: upsdie-down camera
            x: points to the left
            y: points upwards
            z: points into the viewing direction
        Args:
            fov_xy: field of view in degrees (x and y directions); list of length 2
            W: width of image; int
            H: height of image; int
        Returns:
            directions: ray directions; numpy array of shape (H*W, 3)
        """
        fov_xy = np.deg2rad(fov_xy) # (2,)
        num_pts = np.array([W, H]) # (2,)
        
        fov_cells = np.deg2rad(fov_xy) / num_pts
        angle_max = fov_cells * (num_pts - 1) / 2
        angle_min = - angle_max
        
        angles_x = np.linspace(angle_max[0], angle_min[0], num_pts[0]) # (W,)
        angles_y = np.linspace(angle_max[1], angle_min[1], num_pts[1]) # (H,)
        angles_x, angles_y = np.meshgrid(angles_x, angles_y, indexing="xy") # (H,W), (H,W)
        angles_x = angles_x.flatten() # (H*W,)
        angles_y = angles_y.flatten() # (H*W,)
        
        x = np.sin(angles_x) # (H*W,)
        y = np.sin(angles_y) # (H*W,)
        z = np.cos(angles_x) * np.cos(angles_y) # (H*W,)
        directions = np.stack((x, y, z), axis=1) # (H*W, 3)
        return directions
    
    def cameraDirections(
        self,
        fx:float,
        fy:float,
        cx:float,
        cy:float,
        W:int,
        H:int,
    ):
        """
        Calculate directions given focal lengths of a camera.
        Coordinate system: upsdie-down camera
            x: points to the left
            y: points upwards
            z: points into the viewing direction
        Args:
            fx: focal length in x direction; float
            fy: focal length in y direction; float
            cx: center of projection in x direction; float
            cy: center of projection in y direction; float
            W: width of image; int
            H: height of image; int
        Returns:
            directions: ray directions; numpy array of shape (H*W, 3)
        """
        us, vs = np.meshgrid(np.arange(W), np.arange(H), indexing="xy") # (H,W), (H,W)
        dir_x = (us - cx + 0.5) / fx # (H,W)
        dir_y = (vs - cy + 0.5) / fy # (H,W)
        dir_z = np.ones_like(us) # (H,W)
        
        directions = np.stack((dir_x, dir_y, dir_z), axis=2) # (H,W,3)
        directions /= np.linalg.norm(directions, axis=2, keepdims=True) # (H,W,3)
        return directions.reshape(-1, 3) # (H*W, 3)
        
        
class PCLCreatorUSS(PCLCreator):
    def __init__(
        self,
    ):
        super().__init__()
        
        self.W = 64
        self.H = 64
        self.directions = self.fovDirections(
            fov_xy=[50, 40],
            W=self.W,
            H=self.H,
        )
        
    def meas2depth(
        self,
        meas:float,
    ):
        """
        Convert depth measurments to meters and filter false measurments.
        Args:
            meas: depth measurments; float
        Returns:
            depth: depth measurments; np.array of floats (H, W)
        """
        if meas >= 50000:
            meas = 0.0
        depth = meas / 5000
        return depth * np.ones((self.H, self.W))
    
    
class PCLCreatorToF(PCLCreator):
    def __init__(self):
        super().__init__()
        
        self.directions = self.fovDirections(
            fov_xy=[45, 45],
            W=8,
            H=8,
        )
        
    def _meas2depth(
        self,
        meas:float,
    ):
        """
        Convert depth measurments to meters and correct reference frame.
        Args:
            meas: depth measurments; float
        Returns:
            depth: depth measurments; float
        """
        meas = np.array(meas, dtype=np.float32)
        depth = 0.001 * meas
        
        depth = depth.reshape(8, 8)
        depth = depth[:, ::-1].T
        return depth
    

class PCLCreatorRS(PCLCreator):
    def __init__(
        self,
    ):
        super().__init__()


    def _meas2depth(
        self,
        meas:float,
    ):
        """
        Convert depth measurments to meters and correct reference frame.
        Args:
            meas: depth measurments; float
        Returns:
            depth: depth measurments; float
        """
        meas = np.array(meas, dtype=np.float32)
        depth = 0.001 * meas
        return depth

if __name__ == '__main__':
    pass