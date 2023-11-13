import numpy as np
import os
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from abc import ABC, abstractmethod

from robotathome import RobotAtHome
from args.args import Args

class SceneBase():
    def __init__(
        self, 
        args:Args
    ):
        """
        Class to handle robot@home2 scenes. The scene's point cloud serves as ground truth.
        Args:
            args: arguments; Args object
        """ 
        self.args = args

        # scene transformation variables: world (in meters) -> cube ([-0.5,0.5]**3) coordinate system
        self.w2c_params = {
            "defined": False,
            "shift": None,
            "scale": None,
            "cube_min": -0.5,
            "cube_max": 0.5,
            "scale_margin": 1.05,
        }

    @abstractmethod
    def _defineParams(self):
        pass
    
    def w2c(
        self, 
        pos, 
        only_scale:bool=False, 
        copy:bool=True
    ):
        """
        Transformation from world (in meters) to cube ([cube_min,cube_max]**3) coordinate system.
        Args:
            pos: position to scale and shift; tensor or array of shape (N, 2/3)
            only_scale: if True, only scale position and do not shift; bool
            copy: if True, copy pos before transformation; bool
        Returns:
            pos: scaled and shifted position; tensor or array of shape (N, 2/3)
        """
        if copy:
            pos = np.copy(pos)
        
        if not self.w2c_params["defined"]:
            self._defineParams()

        if not only_scale:
            pos -= self.w2c_params["shift"][:pos.shape[1]]
        pos /=  self.w2c_params["scale"]
        return pos
    
    def c2w(
        self, 
        pos, 
        only_scale:bool=False, 
        copy:bool=True
    ):
        """
        Transformation from cube ([cube_min,cube_max]**3) to world (in meters) coordinate system.
        Args:
            pos: position to scale and shift; tensor or array of shape (N, 2/3)
            only_scale: if True, only scale position and do not shift; bool
            copy: if True, copy pos before transformation; bool
        Returns:
            pos: scaled and shifted position; tensor or array of shape (N, 2/3)
        """
        if copy:
            pos = np.copy(pos)

        if not self.w2c_params["defined"]:
            self._defineParams()

        pos *=  self.w2c_params["scale"]
        if not only_scale:
            pos += self.w2c_params["shift"][:pos.shape[1]]
        return pos
    
    def c2idx(
        self, 
        pos, 
        res:int
    ):
        """
        Transformation from cube coordinate system ([cube_min,cube_max]**2) to slice map indices ([0,res-1]**2).
        Args:
            pos: position to transform; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
        """
        map_idxs = (res - 1) * (pos - self.w2c_params["cube_min"]) \
                / (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]) # (N, x y)
        return np.clip(map_idxs.round().astype(int), 0, res-1) # convert to int
    
    def idx2c(
        self, 
        map_idxs, 
        res:int
    ):
        """
        Transformation from slice map indices ([0,res-1]**2) to cube coordinate system ([cube_min,cube_max]**2).
        Args:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            pos: position in cube coordinate system; tensor or array of shape (N, 2)
        """
        pos = (map_idxs * (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]) / (res - 1)) \
                + self.w2c_params["cube_min"] # (N, x y)
        return pos
    
    def w2idx(
        self, 
        pos, 
        res:int
    ):
        """
        Transformation from world (in meters) to slice map indices ([0,res-1]**2).
        Args:
            pos: position to transform; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
        """
        pos_c = self.w2c(pos=pos)
        return self.c2idx(pos=pos_c, res=res)
    
    def idx2w(
        self, 
        map_idxs, 
        res:int
    ):
        """
        Transformation from slice map indices ([0,res-1]**2) to world (in meters).
        Args:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            pos_w3: position in world coordinate system; tensor or array of shape (N, 2)
        """
        pos_c = self.idx2c(map_idxs=map_idxs, res=res)
        return self.c2w(pos=pos_c)
    
    def depth2pos(
        self, 
        rays_o:np.array, 
        scan_depth:np.array, 
        scan_angles:np.array
    ):
        """
        Convert depth values to positions.
        Args:
            rays_o: origin of scan rays in cube coordinates; numpy array of shape (N, 2/3)
            scan_depth: distance to closest point in scene at rays_o; numpy array of shape (N,)
            scan_angles: angles of scan rays; numpy array of shape (N,)
        Returns:
            pos: position in cube coordinate system; numpy array of shape (N, 2)
        """
        val_idxs = np.where(~np.isnan(scan_depth))[0]
        
        pos = np.full((scan_depth.shape[0], 2), np.nan) # (N, 2)
        pos[val_idxs] = np.stack((scan_depth[val_idxs] * np.cos(scan_angles[val_idxs]), 
                                  scan_depth[val_idxs] * np.sin(scan_angles[val_idxs])), axis=1)
        pos[val_idxs] += rays_o[val_idxs,:2] # (N, 2)
        return pos

    def _calcScanRays(
        self, 
        res:int, 
        rays_o:np.array, 
        rays_d:np.array=None, 
        angular_range:list=[-np.pi,np.pi]
    ):
        """
        Calculate scan rays in cube coordinate system. If rays_d is not given (None), the angles are linearly spaced 
        in the angular range.
        Args:
            res: size of slice map; int
            rays_o: origin of scan rays in cube coordinates; numpy array of shape (N, 3)
            rays_d: direction of scan rays; numpy array of shape (N, 3)
            angular_range: range of scan angles ([min, max] where min is inclusive and max is exclusive); list of two floats
        Returns:
            scan_rays_c: scan rays in cube coordinate system; numpy array of shape (N*M, 2)
            scan_angles: angles of scan rays; numpy array of shape (N,)
        """
        rays_o = np.copy(rays_o[:,:2]) # (N, 2)

        # if rays_d is None, linearly space angles between in angular_range
        # else, use given ray directions
        if rays_d is None:
            scan_angles = np.linspace(angular_range[0], angular_range[1], rays_o.shape[0], endpoint=False) # (N,)
        else:
            scan_angles = np.arctan2(rays_d[:,1], rays_d[:,0]) # (N,)

        # get relative scan rays
        M = np.ceil(np.sqrt(2*res**2)).astype(int)
        r_points = np.linspace(0, (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]), M) # (M,)
        m_points, m_angles = np.meshgrid(r_points, scan_angles, indexing="xy") # (N, M)
        r_x = m_points * np.cos(m_angles) # (N, M)
        r_y = m_points * np.sin(m_angles) # (N, M)
        r_c = np.stack((r_x.flatten(), r_y.flatten()), axis=1) # (N*M, 2)

        # get absolute scan rays in cube coordinate system
        rays_o = np.repeat(rays_o, M, axis=0) # (N*M, 2)
        scan_rays_c = r_c + rays_o # (N*M, 2)
        return scan_rays_c, scan_angles
