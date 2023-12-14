#!/usr/bin/env python

from rosbag import Bag
import numpy as np

from rosbag_wrapper import RosbagWrapper
from matplotlib import pyplot as plt


class RosbagSync(RosbagWrapper):
    def __init__(
        self,
        bag_path:str,
    ) -> None:
        super().__init__(
            bag_path=bag_path,
        )
        
    def getTimeRS(
        self,
    ):
        meass, times_uss = self.read(
            topic="/USS1",
        )
    
    
    def getTimeUSS(
        self,
        times_rs_cor:np.array,
    ):
        meass, times_uss = self.read(
            topic="/USS1",
        )
        # convert measurements to depths
        meass[meass >= 50000] = 0.0
        depths = meass / 5000

        # convert depths to time delays
        delays = depths / 343.0
        
        # correct uss times
        times_uss -= delays
        
        times_uss_rep = np.tile(times_uss, (times_rs_cor.shape[0], 1))
        times_rs_cor_rep = np.tile(times_rs_cor, (times_uss.shape[0], 1)).T
        idxs = np.argmin((times_uss_rep > times_rs_cor_rep), axis=0)
        times_uss_cor = times_uss[idxs]
        
        return times_uss_cor, times_uss
    
    def compareTimes(
        self,
    ):
        _, times_uss = self.read(
            topic="/USS3",
        )
        _, times_cam = self.read(
            topic="/CAM3/color/image_raw",
        )
        
        print(f"times_uss: max: {np.max(times_uss)}, min: {np.min(times_uss)}")
        print(f"times_cam: max: {np.max(times_cam)}, min: {np.min(times_cam)}")
        print(f"times_uss: freq: {1.0 / np.mean(np.diff(times_uss))}")
        print(f"times_cam: freq: {1.0 / np.mean(np.diff(times_cam))}")
        
        time_start = np.min([times_uss[0], times_cam[0]])
        times_uss -= time_start
        times_cam -= time_start
        plt.scatter(times_uss, np.zeros_like(times_uss), label="USS")
        plt.scatter(times_cam, np.ones_like(times_cam), label="CAM")
        plt.legend()
        plt.xlim([0, 5])
        plt.show()
        




def main():

    bag = RosbagSync(
        bag_path="/home/spadmin/catkin_ws_ngp/data/DataSync/test.bag",
    )
    bag.compareTimes()

if __name__ == "__main__":
    main()