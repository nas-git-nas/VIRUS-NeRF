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
    
    def getSliceScan(self, points, map_res, angular_res, height_tolerance=0.1, angular_range=[-np.pi,np.pi], points_in_world_coord=True):
        """
        Get a scan of the scene at points. The scan is similar to a horizontal 2D LiDAR scan and represents
        the distance to the closest point in the scene in the direction of the scan.
        Args:
            points: points to scan in world coordinates; numpy array of shape (N, 3)
            map_res: size of slice map; int
            angular_res: size of scan / nb. angles per scan; int
            height_tolerance: tolerance for height in world coordinate system; float
            angular_range: range of scan angles ([min, max] where min is inclusive and max is exclusive); list of two floats
        Returns:
            slice_scans: scan of the scene at points; numpy array of shape (map_res, map_res)
        """
        # transform points to cube coordinate system
        if points_in_world_coord:
            points = self.w2cTransformation(pos=points, copy=True) # (N, 3)

        # get scan rays
        r_angles = np.linspace(angular_range[0], angular_range[1], angular_res, endpoint=False) # (angular_res,)
        r_points = np.linspace(0, (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]),
                                 np.ceil(np.sqrt(2*map_res**2)).astype(int)) # (nb_points,)
        r_points, r_angles = np.meshgrid(r_points, r_angles, indexing="xy") # (angular_res, nb_points)
        r_x = r_points * np.cos(r_angles) # (angular_res, nb_points)
        r_y = r_points * np.sin(r_angles) # (angular_res, nb_points)
        r_c = np.stack((r_x.flatten(), r_y.flatten()), axis=1) # (angular_res*nb_points, 2)

        # get slice maps for unique heights
        slice_maps = {}
        for height in np.unique(points[:,2]):
            slice_map = self.getSliceMap(height, map_res, height_tolerance=height_tolerance, height_in_world_coord=False) # (map_res, map_res)
            slice_maps[height] = slice_map

        # get slice scans
        slice_scans = np.zeros((points.shape[0], map_res, map_res)) # (map_res, map_res)
        for i in range(points.shape[0]):
            # get occupancy status of points on rays
            rays_c = r_c + points[i,:2] # (angular_res*nb_points, 2)
            rays_idxs = self.c2idxTransformation(pos=rays_c, res=map_res) # (angular_res*nb_points, 2)
            rays_occ = slice_maps[points[i,2]][rays_idxs[:,0], rays_idxs[:,1]] # (angular_res*nb_points,)

            rays_idxs = rays_idxs.reshape((angular_res, -1, 2)) # (angular_res, nb_points, 2)
            rays_occ = rays_occ.reshape((angular_res, -1)) # (angular_res, nb_points)

            # get closest occupied point on rays
            angle_idxs, point_idxs = np.where(rays_occ > 0) 
            angle_idxs, id = np.unique(angle_idxs, return_index=True) # (occ_points,)
            point_idxs = point_idxs[id] # (occ_points,)

            rays_idxs = rays_idxs[angle_idxs, point_idxs] # (occ_points, 2)
            slice_scans[i, rays_idxs[:,0], rays_idxs[:,1]] = 1

        return slice_scans    
    
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

    # get slice map and scan
    points_w = np.array([[0,0,0.5],
                         [2.5,0,0.5],
                         [0.5,-2,0.5]])
    
    res = 256
    slice_map = rh_scene.getSliceMap(height=points_w[0,2], res=res)
    slice_scans = rh_scene.getSliceScan(points=np.copy(points_w), map_res=res, angular_res=256, angular_range=[-np.pi/2,np.pi])

    fig, axes = plt.subplots(ncols=3, nrows=slice_scans.shape[0], figsize=(12,8))
    extent = rh_scene.c2wTransformation(pos=np.array([[-0.5,-0.5],[0.5,0.5]]), copy=False)
    extent = extent.T.flatten()

    for i in range(slice_scans.shape[0]):   
        comb_map = slice_map + 2*slice_scans[i]
        # score = np.sum(slice_map * slice_scans[i]) / np.sum(slice_scans[i])

        ax = axes[i,0]
        ax.imshow(slice_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
        ax.scatter(points_w[i,0], points_w[i,1], c='r', s=10)
        if i == 0:
            ax.set_title(f'Slice Map')
        if i == slice_scans.shape[0]-1:
            ax.set_xlabel(f'x [m]')
        ax.set_ylabel(f'y [m]')

        ax = axes[i,1]
        ax.imshow(2*slice_scans[i].T,origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
        ax.scatter(points_w[i,0], points_w[i,1], c='r', s=10)
        if i == 0:
            ax.set_title(f'Slice Scan')
        if i == slice_scans.shape[0]-1:
            ax.set_xlabel(f'x [m]')

        ax = axes[i,2]
        ax.imshow(comb_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
        ax.scatter(points_w[i,0], points_w[i,1], c='r', s=10)
        if i == 0:
            ax.set_title(f'Combined Map')
        if i == slice_scans.shape[0]-1:
            ax.set_xlabel(f'x [m]')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_RobotAtHomeScene()