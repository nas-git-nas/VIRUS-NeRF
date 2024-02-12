import matplotlib.pyplot as plt
import numpy as np
import os
import sys
 
sys.path.insert(0, os.getcwd())
from ETHZ_experiments.catkin_ws.src.sensors.src.pcl_tools.pcl_loader import PCLLoader


def plot_pointclouds(
    pointcloud_dir: str,
):
    
    pcl_loader = PCLLoader(
        data_dir=pointcloud_dir,
    )

    files = pcl_loader.getFiles(
        pcl_dir="nerf_pcl",
    )

    xyzs = []
    for f in files:
        xyz = pcl_loader.loadPCL(
            filename=os.path.join("nerf_pcl", f),
        )
        xyzs.append(xyz)

    xyzs = np.array(xyzs)
    xyzs = xyzs.reshape(-1, 3)

    # plot point cloud
    plt.scatter(xyzs[:,0], xyzs[:,1], s=0.1, c=np.arange(xyzs.shape[0]), cmap='jet')
    plt.show()    

def main():
    pointcloud_dir = "C:/Users/nicol/OneDrive/Documents/EPFL/11MasterThesis/Code/ngp_fusion/results/ETHZ/pointclouds/20240209_081913_office_pointclouds"
    plot_pointclouds(
        pointcloud_dir=pointcloud_dir,
    )



if __name__ == '__main__':
    main()