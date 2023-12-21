import numpy as np 
# import open3d as o3d
from pypcd4 import PointCloud
import os


class PCLLoader():
    def __init__(
        self,
        data_dir:str,
        pcl_dir:str,
    ) -> None:
        self.data_dir = data_dir
        self.pcl_dir = pcl_dir
        
    def loadLatestPCL(
        self,
    ):
        """
        Load the latest point cloud in time from the maps directory.
        Returns:
            xyz: point cloud; np.array of shape (N, 3)
        """
        # load all time stamps of the point clouds assuming that the filenames are the time stamps
        filenames = self._getFiles()
        times = self._filename2time(
            filenames=filenames
        )
        
        # determine the latest point cloud
        idx_max = np.argmax(times)
        
        return self.loadPCL(
            filename=filenames[idx_max]
        )
    
    def loadPCL(
        self,
        filename:str,
    ):
        """
        Load a point cloud from the maps directory.
        Args:
            filename: filename of the point cloud; str
        Returns:
            xyz: point cloud; np.array of shape (N, 3)
        """
        # pcd = o3d.io.read_point_cloud(os.path.join(self.data_dir, self.pcl_dir, filename))
        pc = PointCloud.from_path(os.path.join(self.data_dir, self.pcl_dir, filename))
        # xyz = np.asarray(pcd.points)
        xyz = pc.numpy(
            fields=["x", "y", "z"],
        )
        return xyz
    
    def renamePCL(
        self,
        prefix:str,
    ):
        filenames = self._getFiles()
        times = self._filename2time(
            filenames=filenames
        )
        idxs = np.argsort(times)
        filenames = [filenames[idx] for idx in idxs]
        
        for i, f in enumerate(filenames):
            os.rename(os.path.join(self.data_dir, self.pcl_dir, f), os.path.join(self.data_dir, self.pcl_dir, prefix + str(i) + ".pcd"))
    
    def _filename2time(
        self,
        filenames:list,
    ):
        """
        Convert filenames to time stamps.
        Args:
            filename: filename of the point cloud; str
        Returns:
            time: time of the point cloud; float
        """
        return [float(t[:-4]) for t in filenames]
        
    def _getFiles(
        self,
    ):
        """
        Get all files in the maps directory.
        Returns:
            filenames: list of files in the maps directory; list of str
        """
        maps_dir = os.path.join(self.data_dir, self.pcl_dir)
        return [f for f in os.listdir(maps_dir) if os.path.isfile(os.path.join(maps_dir, f))]
    
    
    
def test_pcl_loader():
    data_dir = "/home/spadmin/catkin_ws_ngp/data/test"
    
    pcl_loader = PCLLoader(
        data_dir=data_dir,
        pcl_dir="maps",
    )
    xyz = pcl_loader.loadLatestPCL()
    
    print(xyz.shape)
    print(xyz[:10])    
    
    pcl_loader.renamePCL(
        prefix="full"
    )   
    
def test_rename():
    data_dir = "/home/spadmin/catkin_ws_ngp/data/test"
    
    pcl_loader = PCLLoader(
        data_dir=data_dir,
        pcl_dir="lidar_scans",
    )
    pcl_loader.renamePCL(
        prefix="full"
    )
    
if __name__ == "__main__":
    # test_pcl_loader()
    test_rename()