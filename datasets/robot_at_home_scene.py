import numpy as np
import os
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from robotathome import RobotAtHome

class RobotAtHomeScene():
    def __init__(self, rh:RobotAtHome, rh_location_names:dict):
        """
        Class to handle robot@home2 scenes. The scene's point cloud serves as ground truth.
        Args:
            rh: robot@home2 database; RobotAtHome object
            rh_location_names: dictionary of location names; dict
        """ 
        self.rh = rh
        self.rh_location_names = rh_location_names

        self._point_cloud = self.__loadPointCloud()

        # scene transformation variables: world (in meters) -> cube ([-0.5,0.5]**3) coordinate system
        self.w2c_params = {
            "defined": False,
            "shift": None,
            "scale": None,
            "cube_min": -0.5,
            "cube_max": 0.5,
            "scale_margin": 1.05,
        }

    def getPointCloud(self):
        """
        Get point cloud of scene.
        Returns:
            point cloud of scene; numpy array (N, x y z R B G)
        """       
        return self._point_cloud
    
    def getSliceMap(self, height, res, height_tolerance=0.1, height_in_world_coord=True):
        """
        Get a map of the scene of size (res,res). The map is a slice of the scene 
        at height [height-height_tolerance, height+height_tolerance].
        Args:
            height: height of the slice in world coordinates; float
            res: size of slice map; int
            height_tolerance: tolerance for height in world coordinates; float
            height_in_world_coord: if True, height is given in world otherwise in cube coordinates; bool
        Returns:
            slice_map: slice of the scene; array of shape (res, res)
        """
        slice_map = np.zeros((res, res)) # (x y)
        point_cloud = np.copy(self._point_cloud[:,:3]) # (N, x y z)

        if not height_in_world_coord:
            height = self.c2wTransformation(pos=np.array([[0,0,height]]))[0,2]

        # extract points in slice [height-height_tolerance, height+height_tolerance]
        point_idxs = np.where((point_cloud[:,2] >= height-height_tolerance) & (point_cloud[:,2] <= height+height_tolerance))[0] # (M,)
        points = point_cloud[point_idxs,:2] # (M, x y z)  

        # convert points to slice map indices ([0,res-1]**2)
        map_idxs = self.w2idxTransformation(pos=points, res=res) # (M, x y)

        # fill slice map
        slice_map[map_idxs[:,0], map_idxs[:,1]] = 1

        return slice_map
    
    def getSliceScan(self, res, rays_o, rays_d=None, height_tolerance=0.1, angular_range=[-np.pi,np.pi], rays_o_in_world_coord=True):
        """
        Get a scan of the scene at points. The scan is similar to a horizontal 2D LiDAR scan and represents
        the distance to the closest point in the scene in the direction of the scan.
        Args:
            points: points to scan in world coordinates; numpy array of shape (N, 3)
            res: size of slice map; int
            angular_res: size of scan / nb. angles per scan; int
            height_tolerance: tolerance for height in world coordinate system; float
            angular_range: range of scan angles ([min, max] where min is inclusive and max is exclusive); list of two floats
            rays_o_in_world_coord: if True, rays_o is given in world otherwise in cube coordinates; bool
        Returns:
            scan_map: scan of the scene at rays_o; numpy array of shape (res, res)
            scan_depth: distance to closest point in scene at rays_o; numpy array of shape (N,)
            scan_ray_angles: angles of scan rays; numpy array of shape (N,)
        """
        if rays_o_in_world_coord:
            rays_o = self.w2cTransformation(pos=rays_o, copy=True) # (N, 2)

        # calculate scan rays in cube coordinate system
        scan_rays_c, scan_angles = self.__calcScanRays(res, rays_o=rays_o, rays_d=rays_d, angular_range=angular_range) # (N*M, 2)

        # verify that all height values are inside tolerance
        height_mean = np.mean(rays_o[:,2])
        if np.any(np.abs(rays_o[:,2] - height_mean) > height_tolerance):
            print(f"ERROR: height values of rays_o are not inside tolerance of {height_tolerance}!" \
                  + f"mean height: {height_mean}, min height: {np.min(rays_o[:,2])}, max height: {np.max(rays_o[:,2])}")

        # get slice map for mean height
        slice_map = self.getSliceMap(height_mean, res, height_tolerance=height_tolerance, height_in_world_coord=False) # (res, res)

        # get occupancy status of points on rays
        scan_rays_idxs = self.c2idxTransformation(pos=scan_rays_c, res=res) # (N*M, 2)
        scan_rays_occ = slice_map[scan_rays_idxs[:,0], scan_rays_idxs[:,1]] # (N*M,)

        scan_rays_idxs = scan_rays_idxs.reshape((rays_o.shape[0], -1, 2)) # (N, M, 2)
        scan_rays_occ = scan_rays_occ.reshape((rays_o.shape[0], -1)) # (N, M)

        # get closest occupied point on rays
        angle_idxs, point_idxs = np.where(scan_rays_occ > 0) # (occ_points,)
        angle_idxs, unique_ixds = np.unique(angle_idxs, return_index=True) # (closest_occ_points,)
        point_idxs = point_idxs[unique_ixds] # (closest_occ_points,)
        scan_rays_closest_idxs = scan_rays_idxs[angle_idxs, point_idxs] # (closest_occ_points, 2)
        
        # create scan map where all closest occupied points on ray are set to 1
        scan_map = np.zeros((res, res)) # (res, res)
        scan_map[scan_rays_closest_idxs[:,0], scan_rays_closest_idxs[:,1]] = 1

        # create scan depth that represents the distance in cube coordinates to the closest occupied point on ray
        scan_depth = np.full(rays_o.shape[0], np.nan) # (N,)
        scan_rays_closest_c = self.idx2cTransformation(map_idxs=scan_rays_closest_idxs, res=res) # (closest_occ_points, 2)
        scan_depth[angle_idxs] = np.linalg.norm(scan_rays_closest_c - rays_o[angle_idxs,:2], axis=1) # (closest_occ_points,)

        # print(f"number of nan values in scan_depth: {np.sum(np.isnan(scan_depth))}")

        # # verify scan_rays_closest_c
        # s_r_cl_idxs = self.c2idxTransformation(pos=scan_rays_closest_c, res=res) # (closest_occ_points, 2)
        # test_map = np.zeros((res, res)) # (res, res)
        # test_map[s_r_cl_idxs[:,0], s_r_cl_idxs[:,1]] = 1
        # if not np.allclose(test_map, scan_map):
        #     print("ERROR: scan_map and test_map are not equal!")

        # scan_depth_pos = self.convertDepth2Pos(rays_o=rays_o, scan_depth=scan_depth, scan_angles=scan_angles) # (N, 2)
        # # not_nan_idxs = np.where(~np.isnan(scan_depth))[0]
        # # scan_depth_pos = np.stack((scan_depth[not_nan_idxs] * np.cos(scan_angles[not_nan_idxs]), 
        # #                            scan_depth[not_nan_idxs] * np.sin(scan_angles[not_nan_idxs])), axis=1) # (N, 2)
        # # scan_depth_pos += rays_o[not_nan_idxs,:2] # (N, 2)
        # scan_depth_pos_idxs = self.c2idxTransformation(pos=scan_depth_pos, res=res) # (N, 2)
        # test_map = np.zeros((res, res)) # (res, res)
        # test_map[scan_depth_pos_idxs[:,0], scan_depth_pos_idxs[:,1]] = 1
        # if not np.allclose(test_map, scan_map):
        #     print("ERROR: scan_map and test_map 2 are not equal!")

        #     fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,8))

        #     ax = axes[0]
        #     ax.imshow(scan_map.T, origin='lower', cmap='viridis', extent=[-0.5,0.5,-0.5,0.5], vmin=0, vmax=np.max(scan_map))
        #     ax.set_title(f'Scan Map')
        #     ax.set_xlabel(f'x [m]')
        #     ax.set_ylabel(f'y [m]')
        #     for i in range(0, scan_depth_pos.shape[0], 10):
        #         ax.plot([rays_o[i,0], scan_depth_pos[i,0]], [rays_o[i,1], scan_depth_pos[i,1]], c='r', linewidth=0.5)

        #     ax = axes[1]
        #     ax.imshow(test_map.T, origin='lower', cmap='viridis', extent=[-0.5,0.5,-0.5,0.5], vmin=0, vmax=np.max(test_map))
        #     ax.set_title(f'Test Map')
        #     ax.set_xlabel(f'x [m]')
        #     ax.set_ylabel(f'y [m]')
        #     for i in range(0, scan_depth_pos.shape[0], 10):
        #         ax.plot([rays_o[i,0], scan_depth_pos[i,0]], [rays_o[i,1], scan_depth_pos[i,1]], c='r', linewidth=0.5)

        #     ax = axes[2]
        #     ax.imshow((scan_map-test_map).T, origin='lower', cmap='viridis', extent=[-0.5,0.5,-0.5,0.5], vmin=-1, vmax=1)
        #     ax.set_title(f'Difference Map')
        #     ax.set_xlabel(f'x [m]')
        #     ax.set_ylabel(f'y [m]')

        #     plt.tight_layout()
        #     plt.show()

        return scan_map, scan_depth, scan_angles
    
    def w2cTransformation(self, pos, only_scale=False, copy=True):
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
            self.__w2cParams()

        if not only_scale:
            pos -= self.w2c_params["shift"][:pos.shape[1]]
        pos /=  self.w2c_params["scale"]
        return pos
    
    def c2wTransformation(self, pos, only_scale=False, copy=True):
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
            self.__w2cParams()

        pos *=  self.w2c_params["scale"]
        if not only_scale:
            pos += self.w2c_params["shift"][:pos.shape[1]]
        return pos
    
    def c2idxTransformation(self, pos, res):
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
    
    def idx2cTransformation(self, map_idxs, res):
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
    
    def w2idxTransformation(self, pos, res):
        """
        Transformation from world (in meters) to slice map indices ([0,res-1]**2).
        Args:
            pos: position to transform; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
        """
        pos_c = self.w2cTransformation(pos=pos)
        return self.c2idxTransformation(pos=pos_c, res=res)
    
    def idx2wTransformation(self, map_idxs, res):
        """
        Transformation from slice map indices ([0,res-1]**2) to world (in meters).
        Args:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            pos_w3: position in world coordinate system; tensor or array of shape (N, 2)
        """
        pos_c = self.idx2cTransformation(map_idxs=map_idxs, res=res)
        return self.c2wTransformation(pos=pos_c)
    
    def convertDepth2Pos(self, rays_o, scan_depth, scan_angles):
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
        pos = np.stack((scan_depth[val_idxs] * np.cos(scan_angles[val_idxs]), 
                                   scan_depth[val_idxs] * np.sin(scan_angles[val_idxs])), axis=1) # (N, 2)
        pos += rays_o[val_idxs,:2] # (N, 2)
        return pos
  
    def __loadPointCloud(self):
        """
        Load scene from robot@home2 database.
        Returns:
            point cloud of scene; numpy array (N, x y z R B G)
        """
        home_session_id = self.rh.name2id(self.rh_location_names["home"]+"-"+self.rh_location_names["home_session"],'hs')
        room_id = self.rh.name2id(self.rh_location_names["home"]+"_"+self.rh_location_names["room"], "r")

        # get scene database of particular room  
        scene =  self.rh.get_scenes().query(f'home_session_id=={home_session_id} & room_id=={room_id}')
    
        # load scene point cloud
        scene_file = scene.scene_file.values[0]
        return np.loadtxt(scene_file, skiprows=6)
    
    def __w2cParams(self):
        """
        Calculate world (in meters) to cube ([cube_min,cube_max]**3) transformation parameters.
        Enlarge the cube with scale_margin s.t. all points are sure to be inside the cube.
        """
        point_cloud = np.copy(self._point_cloud[:,:3]) # (N, x y z)

        # get scene shift and scale
        xyz_min = point_cloud.min(axis=0)
        xyz_max = point_cloud.max(axis=0)
        shift = (xyz_max + xyz_min) / 2
        scale = (xyz_max - xyz_min).max() * self.w2c_params["scale_margin"] \
                / (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]) 

        # set world to cube transformation parameters
        self.w2c_params["defined"] = True
        self.w2c_params["shift"] = shift
        self.w2c_params["scale"] = scale

    def __calcScanRays(self, res, rays_o, rays_d=None, angular_range=[-np.pi,np.pi]):
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

    


def test_RobotAtHomeScene():
    # load dataset
    my_rh_path = '../RobotAtHome2/data'
    my_rgbd_path = os.path.join(my_rh_path, 'files/rgbd')
    my_scene_path = os.path.join(my_rh_path, 'files/scene')
    my_wspc_path = 'results'
    my_db_filename = "rh.db"
    rh = RobotAtHome(rh_path=my_rh_path, rgbd_path=my_rgbd_path, scene_path=my_scene_path, wspc_path=my_wspc_path, db_filename=my_db_filename)

    # load scene
    rh_location_names = {
        "session": "session_2",
        "home": "anto",
        "room": "livingroom1",
        "subsession": "subsession_1",
        "home_session": "s1",
    }
    rh_scene = RobotAtHomeScene(rh, rh_location_names)

    # get slice map
    res = 256
    res_angular = 256
    rays_o_w = np.array([[0,0,0.5]])
    rays_o_w = np.repeat(rays_o_w, res_angular, axis=0)
    slice_map = rh_scene.getSliceMap(height=rays_o_w[0,2], res=res)

    # get slice scan
    rays_d_is_given = True
    if rays_d_is_given:
        rays_o_w2 = np.array([[0,-2,0.5]])
        rays_o_w2 = np.repeat(rays_o_w2, res_angular, axis=0)
        angles_d1 = np.linspace(-np.pi/3, np.pi/3, res_angular, endpoint=False)
        rays_d1 = np.stack((np.cos(angles_d1), np.sin(angles_d1), np.zeros_like(angles_d1)), axis=1)
        angles_d2 = np.linspace(0, 2*np.pi/3, res_angular, endpoint=False)
        rays_d2 = np.stack((np.cos(angles_d2), np.sin(angles_d2), np.zeros_like(angles_d2)), axis=1)

        rays_o_w = np.concatenate((rays_o_w, rays_o_w2), axis=0)
        rays_d = np.concatenate((rays_d1, rays_d2), axis=0)

        scan_map, scan_depth_c, scan_angles = rh_scene.getSliceScan(res=res, rays_o=rays_o_w, rays_d=rays_d, rays_o_in_world_coord=True)
    else:
        scan_map, scan_depth_c, scan_angles = rh_scene.getSliceScan(res=res, rays_o=rays_o_w, rays_d=None, rays_o_in_world_coord=True)

    # convert scan depth to position in world coordinate system
    rays_o_c = rh_scene.w2cTransformation(pos=rays_o_w, copy=True)
    scan_depth_pos_c = rh_scene.convertDepth2Pos(rays_o=rays_o_c, scan_depth=scan_depth_c, scan_angles=scan_angles) # (N, 2)
    scan_depth_pos_w = rh_scene.c2wTransformation(pos=scan_depth_pos_c, copy=True) # (N, 2)

    # plot
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,6))
    extent = rh_scene.c2wTransformation(pos=np.array([[-0.5,-0.5],[0.5,0.5]]), copy=False)
    extent = extent.T.flatten()
    rays_o_w_unique = np.unique(rays_o_w, axis=0)
    comb_map = slice_map + 2*scan_map
    # score = np.sum(slice_map * slice_scans[i]) / np.sum(slice_scans[i])

    ax = axes[0]
    ax.imshow(slice_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
    ax.scatter(rays_o_w_unique[:,0], rays_o_w_unique[:,1], c='w', s=5)
    ax.set_title(f'Map')
    ax.set_xlabel(f'x [m]')
    ax.set_ylabel(f'y [m]')

    ax = axes[1]
    ax.imshow(2*scan_map.T,origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
    for i in range(scan_depth_pos_w.shape[0]):
        ax.plot([rays_o_w[i,0], scan_depth_pos_w[i,0]], [rays_o_w[i,1], scan_depth_pos_w[i,1]], c='w', linewidth=0.1)
    ax.scatter(rays_o_w_unique[:,0], rays_o_w_unique[:,1], c='w', s=5)
    ax.set_title(f'Scan')
    ax.set_xlabel(f'x [m]')
    
    ax = axes[2]
    ax.imshow(comb_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
    for i in range(scan_depth_pos_w.shape[0]):
        ax.plot([rays_o_w[i,0], scan_depth_pos_w[i,0]], [rays_o_w[i,1], scan_depth_pos_w[i,1]], c='w', linewidth=0.1)
    ax.scatter(rays_o_w_unique[:,0], rays_o_w_unique[:,1], c='w', s=5)
    ax.set_title(f'Combined')
    ax.set_xlabel(f'x [m]')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_RobotAtHomeScene()