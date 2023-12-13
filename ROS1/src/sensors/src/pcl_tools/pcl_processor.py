#!/usr/bin/env python
import numpy as np


class PCLProcessor:
    def __init__(
        self,
    ):
        pass
        
    def limitXYZ(
        self,
        xyz:np.array,
        x_lims:list=None,
        y_lims:list=None,
        z_lims:list=None,
    ):
        """
        Limit the pointcloud to a certain range in x, y and z.
        Args:
            xyz: pointcloud; numpy array (N,3)
            x_lims: limits in x; list of length 2
            y_lims: limits in y; list of length 2
            z_lims: limits in z; list of length 2
        Returns:
            xyz: limited pointcloud; numpy array (N,3)
        """
        if x_lims is not None:
            xyz = xyz[np.logical_and(xyz[:,0] >= x_lims[0], xyz[:,0] <= x_lims[1])]
        if y_lims is not None:
            xyz = xyz[np.logical_and(xyz[:,1] >= y_lims[0], xyz[:,1] <= y_lims[1])]
        if z_lims is not None:
            xyz = xyz[np.logical_and(xyz[:,2] >= z_lims[0], xyz[:,2] <= z_lims[1])]
        return xyz
    
    def limitRTP(
        self,
        xyz,
        r_lims:list=None,
        t_lims:list=None,
        p_lims:list=None,
    ):
        """
        Limit the pointcloud to a certain range in radius, theta and phi.
        Args:
            xyz: pointcloud; numpy array (N,3)
            r_lims: limits in radius; list of length 2
            t_lims: limits in theta [degrees]; list of length 2
            p_lims: limits in phi [degrees]; list of length 2
        Returns:
            xyz: limited pointcloud; numpy array (N,3)
        """
        rtp = self._cart2sph(
            xyz=xyz
        )
        
        if r_lims is not None:
            xyz = xyz[np.logical_and(rtp[:,0] >= r_lims[0], rtp[:,0] <= r_lims[1])]
        if t_lims is not None:
            xyz = xyz[np.logical_and(rtp[:,1] >= t_lims[0], rtp[:,1] <= t_lims[1])]
        if p_lims is not None:
            xyz = xyz[np.logical_and(rtp[:,2] >= p_lims[0], rtp[:,2] <= p_lims[1])]
        
        xyz = self._sph2cart(
            rtp=rtp
        )
        return xyz
        
    def offsetDepth(
        self,
        xyz,
        offset,
    ):
        """
        Offsets the depth of a pointcloud assuming sensor is at position (0,0,0).
        Coordinate system: LiDAR
            x: points in the viewing direction
            y: points to the left
            z: points upwards
        Args:
            xyz: pointcloud; numpy array (N,3)
            offset: offset in ray direction; float
        Returns:
            xyz: pointcloud with offset; numpy array (N,3)
        """
        rtp = self._cart2sph(
            xyz=xyz
        )
        
        rtp[:,0] += offset
        
        xyz = self._sph2cart(
            rtp=rtp
        )
        return xyz

    def _cart2sph(
        self,
        xyz
    ):
        """
        Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi).
        Args:
            xyz: points in cartesian coordinates; numpy array (N,3)
        Returns:
            rtp: points in polar coordinates; numpy array (N,3)
        """
        radius = np.linalg.norm(xyz, axis=1)
        theta = np.arctan2(np.sqrt(xyz[:,0]**2 + xyz[:,1]**2), xyz[:,2])
        phi = np.arctan2(xyz[:,1], xyz[:,0])
        return np.stack((radius, theta, phi), axis=1)

    def _sph2cart(
        self,
        rtp,
    ):
        """
        Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z).
        Args:
            xyz: points in polar coordinates; numpy array (N,3)
        Returns:
            rtp: points in cartesian coordinates; numpy array (N,3)
        """
        x = rtp[:,0] * np.cos(rtp[:,2]) * np.sin(rtp[:,1])
        y = rtp[:,0] * np.sin(rtp[:,2]) * np.sin(rtp[:,1])
        z = rtp[:,0] * np.cos(rtp[:,1])
        return np.stack((x, y, z), axis=1)
        

if __name__ == '__main__':
    pass
