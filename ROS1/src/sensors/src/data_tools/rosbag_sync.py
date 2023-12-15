#!/usr/bin/env python

from rosbag import Bag
import numpy as np
import os

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
        
        self.plot_range = [0, 8]
        self.meas_error_max_uss = 0.1
        self.meas_error_max_tof = 0.15
        self.time_error_max_uss = 0.15
        self.time_error_max_tof = 0.05
        
    def syncBag(
        self,
        plot_dir:str=None,
    ):
        
        stack1_sync_paths = self.syncSensorStack(
            stack_id=1,
            plot_dir=plot_dir,
        )
        stack3_sync_paths = self.syncSensorStack(
            stack_id=3,
            plot_dir=plot_dir,
        )
        bag_sync_path = stack1_sync_paths + stack3_sync_paths + [self.bag_path]
        keep_topics = (len(stack1_sync_paths) + len(stack3_sync_paths)) * ["all"] \
                        + [["/CAM1/depth/color/points", "/CAM3/depth/color/points", "/rslidar_points"]]
        delete_ins = (len(stack1_sync_paths) + len(stack3_sync_paths)) * [True] + [False]
        
        self.merge(
            bag_path_out=self.bag_path.replace(".bag", "_sync.bag"),
            bag_paths_ins=bag_sync_path,
            keep_topics=keep_topics,
            delete_ins=delete_ins,
        )
    
    def syncSensorStack(
        self,
        stack_id:int,
        plot_dir:str=None,
    ):
        meass_uss, times_uss = self.read(
            topic="/USS"+str(stack_id),
        )
        meass_tof, times_tof = self.read(
            topic="/TOF"+str(stack_id),
        )
        _, times_rs = self.read(
            topic="/CAM"+str(stack_id)+"/color/image_raw",
        )
        
        axs = [None, None]
        if plot_dir is not None:
            fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 7))
        
        times_closest_uss, mask_uss, axs[0,0], axs[1,0], axs[2,0] = self._findClosestTime(
            times_source=times_uss,
            times_target=times_rs,
            meass_source=meass_uss,
            ax_cor=axs[0,0],
            ax_err_time=axs[1,0],
            ax_err_meas=axs[2,0],
            sensor="USS",
        )
        times_closest_tof, mask_tof, axs[0,1], axs[1,1], axs[2,1] = self._findClosestTime(
            times_source=times_tof,
            times_target=times_rs,
            meass_source=meass_tof,
            ax_cor=axs[0,1],
            ax_err_time=axs[1,1],
            ax_err_meas=axs[2,1],
            sensor="ToF",
        )
        
        mask = mask_uss & mask_tof
        
        bag_path_sync_uss = self.bag_path.replace(".bag", "_sync_uss"+str(stack_id)+".bag")
        self.writeTimeStamp(
            bag_path_sync=bag_path_sync_uss,
            topic_async="/USS"+str(stack_id),
            topic_sync="/USS"+str(stack_id)+"_sync",
            time_async=times_closest_uss[mask],
            time_sync=times_rs[mask],
        )
        
        bag_path_sync_tof = self.bag_path.replace(".bag", "_sync_tof"+str(stack_id)+".bag")
        self.writeTimeStamp(
            bag_path_sync=bag_path_sync_tof,
            topic_async="/TOF"+str(stack_id),
            topic_sync="/TOF"+str(stack_id)+"_sync",
            time_async=times_closest_tof[mask],
            time_sync=times_rs[mask],
        )
        
        bag_path_sync_cam = self.bag_path.replace(".bag", "_sync_rs"+str(stack_id)+".bag")
        self.writeTimeStamp(
            bag_path_sync=bag_path_sync_cam,
            topic_async="/CAM"+str(stack_id)+"/color/image_raw",
            topic_sync="/CAM"+str(stack_id)+"/color/image_raw_sync",
            time_async=times_rs[mask],
            time_sync=times_rs[mask],
        )
        bag_path_sync_list = [bag_path_sync_uss, bag_path_sync_tof, bag_path_sync_cam]
        
        if plot_dir is not None:
            freq_rs = self._calcFreq(
                times=times_rs,
            )
            fig.suptitle(f"Synchronization on RS time stamps (RS freq = {freq_rs:.2f} Hz), keeping {mask.sum()}/{mask.shape[0]} samples")
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            path_splits = self.bag_path.split("/")
            plt.savefig(os.path.join(plot_dir, path_splits[-1].replace(".bag", "") + f"_stack{stack_id}_sync.png"))
            
        return bag_path_sync_list
    
    def _findClosestTime(
        self,
        sensor:str,
        times_source:np.array,
        times_target:np.array,
        meass_source:np.array,
        ax_cor:plt.Axes=None,
        ax_err_time:plt.Axes=None,
        ax_err_meas:plt.Axes=None,
    ):
        """
        Determine closest source measurement to target measurement:
        for every target time find closest source time
        Args:
            times_source: time stamps to adjust to times_target; np.array of floats (N,)
            times_target: reference time stamps; np.array of floats (M,)
            meass_source: measurements to adjust to times_target; np.array of floats (N,)
            sensor: sensor name; str
            ax_cor: axis to plot time correspondence; plt.Axes
            axs_err_time: axis to plot time error; plt.Axes
            axs_err_meas: axis to plot meas error; plt.Axes
        Returns:
            times_source_closest: corresponding time of source to time of target; np.array of ints (M,)
            mask: mask of valid measurements; np.array of bools (M,)
            ax_cor: axis; plt.Axes
            axs_err_time: axis; plt.Axes
            axs_err_meas: axis; plt.Axes
        """
        times_source = np.copy(times_source)
        times_target = np.copy(times_target)
        meass_source = np.copy(meass_source)
        
        times_source_rep = np.tile(times_source, (times_target.shape[0], 1)) # (M, N)
        times_target_rep = np.tile(times_target, (times_source.shape[0], 1)).T # (M, N)
        idxs_sort = np.argsort(np.abs(times_source_rep - times_target_rep), axis=1) # (M, N)
        idxs1 = idxs_sort[:,0] # (M,)
        idxs2 = idxs_sort[:,1] # (M,)
        
        times_source_closest = times_source[idxs1] # (M,)
        
        times_error = np.abs(times_target - times_source_closest) # (M,)
        meass_error = np.abs(meass_source[idxs1] - meass_source[idxs2]) # (M,)
        
        # convert measurement errors to meters
        if sensor == "USS":
            meass_error = meass_error / 5000.0
        elif sensor == "ToF":
            meass_error = np.mean(meass_error, axis=1)
            meass_error = meass_error / 1000.0
        
        # create mask
        if sensor == "USS":
            mask = (meass_error < self.meas_error_max_uss) & (times_error < self.time_error_max_uss)
        elif sensor == "ToF":
            mask = (meass_error < self.meas_error_max_tof) & (times_error < self.time_error_max_tof)
            
        # # if a source measurement was assigned to multiple target measurements, mask all but the closest one
        # idxs_unique, idxs_counts = np.unique(idxs1, return_counts=True)
        # idxs_unique = idxs_unique[idxs_counts > 1]
        # for idx in idxs_unique:
        #     idxs = np.where(idxs1 == idx)[0]
        #     arg_dists = np.argsort(np.abs(times_source[idx] - times_target[idxs]))
        #     idxs = arg_dists[1:]
        #     mask[idxs] = False
        
        if (ax_cor is None) or (ax_err_time is None) or (ax_err_meas is None):
            return times_source_closest, mask, None, None, None
        
        ax_cor = self._plotTimeCorrespondence(
            ax=ax_cor,
            times_source=times_source,
            times_target=times_target,
            mask=mask,
            sensor=sensor,
            idxs=idxs1,
        )
        ax_err_time = self._plotTimeError(
            ax=ax_err_time,
            times_target=times_target,
            times_error=times_error,
            sensor=sensor,
            mask=mask,
        )
        ax_err_meas = self._plotMeasError(
            ax=ax_err_meas,
            times_target=times_target,
            meass_error=meass_error,
            mask=mask,
            sensor=sensor,
        )
        return times_source_closest, mask, ax_cor, ax_err_time, ax_err_meas
    
    def _plotTimeCorrespondence(
        self,
        ax:plt.Axes,
        times_source:np.array,
        times_target:np.array,
        mask:np.array,
        sensor:str,
        idxs:np.array,
    ):        
        """
        Plot closest time.
        Args:
            ax: axis to plot on; plt.Axes
            times_source: time stamps to adjust to times_target; np.array of floats (N,)
            times_target: reference time stamps; np.array of floats (M,)
            mask: mask of valid measurements; np.array of bools (M,)
            sensor: sensor name; str
            idxs: indices of closest time; np.array of ints (M,)
        Returns:
            ax: axis with plot; plt.Axes
        """
        time_start = np.copy(times_target[0])
        times_source -= time_start
        times_target -= time_start
        
        colors_target, colors_source = self._getColors(
            M=times_target.shape[0],
            N=times_source.shape[0],
            idxs=idxs,
        )

        # determine which source sample is used for multiple target samples
        idxs_unique, idxs_inverse, idxs_counts = np.unique(idxs, return_inverse=True, return_counts=True)
        idxs_counts = idxs_counts[idxs_inverse] # (M,)
        mask_star_target = ~(idxs_counts > 1) # (M,)
        
        # convert star mask from target to source space
        mask_star = np.ones_like(times_source, dtype=np.bool_)
        mask_star[idxs] = mask_star_target
        
        ax.scatter(times_source[mask_star], 1.0 * np.ones_like(times_source[mask_star]), label="source", color=colors_source[mask_star])
        ax.scatter(times_source[~mask_star], 1.0 * np.ones_like(times_source[~mask_star]), label="source", color=colors_source[~mask_star], marker="s")
        ax.scatter(times_target[mask], 0.0*np.ones_like(times_target[mask]), label="times error", color=colors_target[mask])
        ax.scatter(times_target[~mask], 0.0*np.ones_like(times_target[~mask]), label="times error", facecolors="none", edgecolors=colors_target[~mask])
        
        ax.set_xlim(self.plot_range)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["RS", sensor])
        
        freq = self._calcFreq(
            times=times_source,
        )
        ax.set_title(f"Time correspondence ({sensor} freq = {freq:.2f} Hz)")
        return ax
    
    def _plotTimeError(
        self,
        ax:plt.Axes,
        times_target:np.array,
        times_error:np.array,
        sensor:str,
        mask:np.array,
    ):        
        """
        Plot closest time.
        Args:
            ax: axis to plot on; plt.Axes
            times_target: reference time stamps; np.array of floats (M,)
            times_error: time error; np.array of floats (M,)
            sensor: sensor name; str
            mask: mask of valid measurements; np.array of bools (M,)
        Returns:
            ax: axis with plot; plt.Axes
        """
        times_target = np.copy(times_target)
        times_error = np.copy(times_error)
        
        colors_target = self._getColors(
            M=times_target.shape[0],
        )
        
        ax.scatter(times_target[mask], times_error[mask], label="times error", color=colors_target[mask])
        ax.scatter(times_target[~mask], times_error[~mask], label="times error", facecolors="none", edgecolors=colors_target[~mask])

        if sensor == "USS":
            error_max = self.time_error_max_uss
        elif sensor == "ToF":
            error_max = self.time_error_max_tof
        ax.hlines(error_max, self.plot_range[0], self.plot_range[1], label="max error", color="k", linestyle="--")

        ax.set_ylabel("error [s]")
        ax.set_xlim(self.plot_range)
        ax.set_title(f"Time error between {sensor} and RS")
        return ax
    
    def _plotMeasError(
        self,
        ax:plt.Axes,
        times_target:np.array,
        meass_error:np.array,
        mask:np.array,
        sensor:str,
    ):        
        """
        Plot closest time.
        Args:
            ax: axis to plot on; plt.Axes
            times_target: reference time stamps; np.array of floats (M,)
            meass_error: measurement error; np.array of floats (M,)
            mask: mask of valid measurements; np.array of bools (M,)
        Returns:
            ax: axis with plot; plt.Axes
        """
        meass_error = np.copy(meass_error)
        times_target = np.copy(times_target)
        
        colors_target = self._getColors(
            M=times_target.shape[0],
        )
        
        ax.scatter(times_target[mask], meass_error[mask], label="meas error", color=colors_target[mask])
        ax.scatter(times_target[~mask], meass_error[~mask], label="meas error", facecolors="none", edgecolors=colors_target[~mask])
        
        if sensor == "USS":
            error_max = self.meas_error_max_uss
        elif sensor == "ToF":
            error_max = self.meas_error_max_tof
        ax.hlines(error_max, self.plot_range[0], self.plot_range[1], label="max error", color="k", linestyle="--")
        
        ax.set_ylabel("error [m]")
        ax.set_xlabel("time [s]")
        ax.set_xlim(self.plot_range)
        ax.set_title(f"Depth error between 2 closest {sensor} samples")
        return ax
    
    def _getColors(
        self,
        M:int,
        N:int=None,
        idxs:np.array=None,
    ):
        """
        Get colors for plotting. If N is None, return only colors for target.
        Args:
            M: size of target; int
            N: size of source; int
            idxs: indices of closest time; np.array of ints (M,)
        Returns:
            colors_target: colors for target; list of str (M,)
            colors_source: colors for source; list of str (N,)
        """
        color_list = ["r", "g", "y", "c", "m", "b"]
        colors_target = np.array([color_list[i % len(color_list)] for i in range(M)])
        
        if N is None:
            return colors_target
        
        colors_source = np.array(["k" for _ in range(N)])
        colors_source[idxs] = colors_target 
        return colors_target, colors_source
        
    def _calcFreq(
        self,
        times:np.array,
    ):
        """
        Calculate average frequency of times.
        Args:
            times: time stamps; np.array of floats (N,)
        Returns:
            freq: average frequency of times; float
        """
        freq = 1.0 / np.mean(np.diff(times))
        return freq
        




def main():

    bag = RosbagSync(
        bag_path="/home/spadmin/catkin_ws_ngp/data/DataSync/office_2.bag",
    )
    bag.syncBag(
        plot_dir="/home/spadmin/catkin_ws_ngp/data/DataSync",
    )

if __name__ == "__main__":
    main()