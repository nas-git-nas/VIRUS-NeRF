import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2 as cv

from args.args import Args
from modules.networks import NGP
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model
from helpers.geometric_fcts import findNearestNeighbour,  createScanPos
from helpers.data_fcts import linInterpolateArray, convolveIgnorNans, dataConverged, downsampleData, smoothIgnoreNans
from helpers.plotting_fcts import combineImgs
from training.metrics_rh import MetricsRH

from modules.occupancy_grid import OccupancyGrid

from training.trainer_base import TrainerBase
from datasets.dataset_base import DatasetBase


class TrainerPlot(TrainerBase):
    def __init__(
        self, 
        hparams_file=None,
        args:Args=None,
        train_dataset:DatasetBase=None,
        test_dataset:DatasetBase=None,
    ):
        TrainerBase.__init__(
            self,
            args=args,
            hparams_file=hparams_file,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

    def _plotOccGrid(
            self,
            step,
    ):
        if not self.args.eval.plot_results:
            return

        if step % self.args.occ_grid.update_interval != 0:
            return

        # calculate mean height in cube, world and occupancy grid coordinates
        height_c = self.train_dataset.getMeanHeight()
        height_w = self.train_dataset.scene.c2w(pos=np.array([[0.0, 0.0, height_c]]), copy=False)[0,2]
        height_o = self.model.occupancy_grid.c2oCoordinates(
            pos_c=height_c,
        )

        occ_3d_grid = self.model.occupancy_grid.getOccupancyCartesianGrid(
            clone=True,
        )
        bin_3d_grid = self.model.occupancy_grid.getBinaryCartesianGrid(
            threshold=self.model.occupancy_grid.threshold,
        )

        # verify that the binary grid is correct
        if self.args.model.debug_mode:
            bitfield = self.model.occupancy_grid.getBitfield(
                clone=True,
            )
            bin_morton_grid = self.model.occupancy_grid.bitfield2morton(
                bin_bitfield=bitfield,
            )
            bin_3d_recovery = self.model.occupancy_grid.morton2cartesian(
                grid_morton=bin_morton_grid,
            )

            if not torch.allclose(bin_3d_grid, bin_3d_recovery):
                self.args.logger.error(f"bin_3d_grid and bin_3d_recovery are not the same")

        # convert from 3D to 2D
        occ_2d_grid = occ_3d_grid[:,:,height_o]
        bin_2d_grid = bin_3d_grid[:,:,height_o]

        # convert from tensor to array
        occ_2d_grid = occ_2d_grid.detach().clone().cpu().numpy()
        bin_2d_grid = bin_2d_grid.detach().clone().cpu().numpy()

        # create density maps
        density_map_gt = self.test_dataset.scene.getSliceMap(
            height=height_w, 
            res=occ_2d_grid.shape[0], 
            height_tolerance=self.args.eval.height_tolerance, 
            height_in_world_coord=True
        ) # (L, L)
        density_map, density_map_thr = self.interfereDensityMap(
            res_map=occ_2d_grid.shape[0],
            height_w=height_w,
            num_avg_heights=1,
            tolerance_w=0.0,
            threshold=0.01 * MAX_SAMPLES / 3**0.5,
        ) # (L, L)

        # plot occupancy grid
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(9,6))
        scale = self.args.model.scale
        extent = self.test_dataset.scene.c2w(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
        extent = extent.T.flatten()

        ax = axes[0,0]
        im = ax.imshow(density_map_gt.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=1)
        ax.set_ylabel(f'y [m]')
        ax.set_title(f'Ground Truth')
        fig.colorbar(im, ax=ax)

        ax = axes[0,1]
        im = ax.imshow(density_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=10 * (0.01 * MAX_SAMPLES / 3**0.5))
        ax.set_ylabel(f'y [m]')
        ax.set_title(f'NeRF density')
        fig.colorbar(im, ax=ax)

        ax = axes[0,2]
        im = ax.imshow(density_map_thr.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'NeRF binary')
        fig.colorbar(im, ax=ax)

        fig.delaxes(axes[1,0])

        ax = axes[1,1]
        im = ax.imshow(occ_2d_grid.T, origin='lower', cmap='viridis', extent=extent, vmin=0, vmax=1)
        ax.set_xlabel(f'x [m]')
        ax.set_ylabel(f'y [m]')
        ax.set_title(f'OccGrid density')
        fig.colorbar(im, ax=ax)

        ax = axes[1,2]
        im = ax.imshow(bin_2d_grid.T, origin='lower', cmap='viridis', extent=extent, vmin=0, vmax=1)
        ax.set_ylabel(f'y [m]')
        ax.set_title(f'OccGrid binary')
        fig.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "occ_grid"+str(step)+".png"))

    @torch.no_grad()
    def _plotEvaluation(
            self,
            data_w:dict,
            metrics_dict:dict,
            num_imgs:int,
            use_relaative_error:bool=False,
    ):
        """
        Plot scan and density maps.
        Args:
            data_w: data dictionary in world coordinates
            metrics_dict: metrics dictionary
            num_imgs: number of images to plot
            use_relaative_error: if True, use relative mnn, else use mnn
        """
        if not self.args.eval.plot_results:
            return
        
        N = num_imgs
        N_down = self.args.eval.num_plot_pts

        # downsample data   
        rays_o_w_nerf, rays_o_w_tof, rays_o_w_uss, rays_o_w_lidar, \
            depth_w_nerf, depth_w_tof, depth_w_uss, depth_w_lidar, \
            depth_w_gt_nerf, depth_w_gt_tof, depth_w_gt_uss, depth_w_gt_lidar, \
            scan_angles_nerf, scan_angles_tof, scan_angles_uss, scan_angles_lidar, \
            nn_dists_nerf, nn_dists_tof, nn_dists_uss, nn_dists_lidar, \
            nn_dists_inv_nerf, nn_dists_inv_tof, nn_dists_inv_uss, nn_dists_inv_lidar = downsampleData(
            datas=[
                data_w['rays_o_nerf'], 
                data_w['rays_o_tof'],
                data_w['rays_o_uss'],
                data_w['rays_o_lidar'],
                data_w['depth_nerf'], 
                data_w['depth_tof'], 
                data_w['depth_uss'], 
                data_w['depth_lidar'],
                data_w['depth_gt_nerf'],
                data_w['depth_gt_tof'],
                data_w['depth_gt_uss'],
                data_w['depth_gt_lidar'],
                data_w['scan_angles_nerf'], 
                data_w['scan_angles_tof'], 
                data_w['scan_angles_uss'], 
                data_w['scan_angles_lidar'],
                metrics_dict['NeRF']['nn_dists'].flatten(), 
                metrics_dict['ToF']['nn_dists'].flatten(),
                metrics_dict['USS']['nn_dists'].flatten(),
                metrics_dict['LiDAR']['nn_dists'].flatten(),
                metrics_dict['NeRF']['nn_dists_inv'].flatten(),
                metrics_dict['ToF']['nn_dists_inv'].flatten(),
                metrics_dict['USS']['nn_dists_inv'].flatten(),
                metrics_dict['LiDAR']['nn_dists_inv'].flatten(),
            ],
            num_imgs=N,
            num_imgs_downsampled=N_down,
        )

        # create scan maps
        scan_maps_nerf = self._scanRays2scanMap(
            rays_o_w=rays_o_w_nerf,
            depth=depth_w_nerf,
            scan_angles=scan_angles_nerf,
            num_imgs=N_down,
        ) # (N, L, L)
        scan_maps_uss = self._scanRays2scanMap(
            rays_o_w=rays_o_w_uss,
            depth=depth_w_uss,
            scan_angles=scan_angles_uss,
            num_imgs=N_down,
        ) # (N, L, L)
        scan_maps_tof = self._scanRays2scanMap(
            rays_o_w=rays_o_w_tof,
            depth=depth_w_tof,
            scan_angles=scan_angles_tof,
            num_imgs=N_down,
        ) # (N, L, L)
        scan_map_lidar = self._scanRays2scanMap(
            rays_o_w=rays_o_w_lidar,
            depth=depth_w_lidar,
            scan_angles=scan_angles_lidar,
            num_imgs=N_down,
        ) # (N, L, L)
        scan_maps_gt_nerf = self._scanRays2scanMap(
            rays_o_w=rays_o_w_nerf,
            depth=depth_w_gt_nerf,
            scan_angles=scan_angles_nerf,
            num_imgs=N_down,
        ) # (N, L, L)
        scan_maps_gt_uss = self._scanRays2scanMap(
            rays_o_w=rays_o_w_uss,
            depth=depth_w_gt_uss,
            scan_angles=scan_angles_uss,
            num_imgs=N_down,
        ) # (N, L, L)
        scan_maps_gt_tof = self._scanRays2scanMap(
            rays_o_w=rays_o_w_tof,
            depth=depth_w_gt_tof,
            scan_angles=scan_angles_tof,
            num_imgs=N_down,
        ) # (N, L, L)
        scan_maps_gt_lidar = self._scanRays2scanMap(
            rays_o_w=rays_o_w_lidar,
            depth=depth_w_gt_lidar,
            scan_angles=scan_angles_lidar,
            num_imgs=N_down,
        ) 
        scan_map_gt = data_w['scan_map_gt'] # (L, L)

        # create scan images by overlaying scan maps with ground truth
        scan_imgs_nerf = []
        scan_imgs_uss = []
        scan_imgs_tof = []
        scan_imgs_lidar = []
        for i in range(N_down):
            img = combineImgs(
                bool_imgs=[scan_map_gt, scan_maps_gt_nerf[i], scan_maps_nerf[i]],
                colors=['grey', 'black', 'orange'],
            )
            scan_imgs_nerf.append(img)
            img = combineImgs(
                bool_imgs=[scan_map_gt, scan_maps_gt_uss[i], scan_maps_uss[i]],
                colors=['grey', 'black', 'blue'],
            )
            scan_imgs_uss.append(img)
            img = combineImgs(
                bool_imgs=[scan_map_gt, scan_maps_gt_tof[i], scan_maps_tof[i]],
                colors=['grey', 'black', 'lime'],
            )
            scan_imgs_tof.append(img)
            img = combineImgs(
                bool_imgs=[scan_map_gt, scan_maps_gt_lidar[i], scan_map_lidar[i]],
                colors=['grey', 'black', 'magenta'],
            )
            scan_imgs_lidar.append(img)

        # dilate scan images for better visualization
        kernel = np.ones((3,3),np.uint8)
        for i in range(N_down):
            scan_imgs_nerf[i] = cv.dilate(scan_imgs_nerf[i].astype(np.uint8), kernel, iterations=1)
            scan_imgs_uss[i] = cv.dilate(scan_imgs_uss[i].astype(np.uint8), kernel, iterations=1)
            scan_imgs_tof[i] = cv.dilate(scan_imgs_tof[i].astype(np.uint8), kernel, iterations=1)
            scan_imgs_lidar[i] = cv.dilate(scan_imgs_lidar[i].astype(np.uint8), kernel, iterations=1)

        # save folder
        save_dir = os.path.join(self.args.save_dir, "maps")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plot
        scale = self.args.model.scale
        extent = self.test_dataset.scene.c2w(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
        extent = extent.T.flatten()
        num_ray_steps = 64
        inlier_thr = 0.1
        max_error_m = 4.0
        bin_size = 0.2
        hist_bins = np.linspace(0, max_error_m, int(max_error_m/bin_size+1))
        
        rays_o_w_nerf = rays_o_w_nerf.reshape(N_down, -1, 3)
        rays_o_w_uss = rays_o_w_uss.reshape(N_down, -1, 3)
        rays_o_w_tof = rays_o_w_tof.reshape(N_down, -1, 3)
        rays_o_w_lidar = rays_o_w_lidar.reshape(N_down, -1, 3)
        depth_w_uss = depth_w_uss.reshape(N_down, -1)
        depth_w_tof = depth_w_tof.reshape(N_down, -1)
        depth_w_nerf = depth_w_nerf.reshape(N_down, -1)
        depth_w_lidar = depth_w_lidar.reshape(N_down, -1)
        depth_w_gt_uss = depth_w_gt_uss.reshape(N_down, -1)
        depth_w_gt_tof = depth_w_gt_tof.reshape(N_down, -1)
        depth_w_gt_nerf = depth_w_gt_nerf.reshape(N_down, -1)
        depth_w_gt_lidar = depth_w_gt_lidar.reshape(N_down, -1)
        scan_angles_uss = scan_angles_uss.reshape(N_down, -1)
        scan_angles_tof = scan_angles_tof.reshape(N_down, -1)
        scan_angles_nerf = scan_angles_nerf.reshape(N_down, -1)
        scan_angles_lidar = scan_angles_lidar.reshape(N_down, -1)
        nn_dists_uss = nn_dists_uss.reshape(N_down, -1)
        nn_dists_tof = nn_dists_tof.reshape(N_down, -1)
        nn_dists_nerf = nn_dists_nerf.reshape(N_down, -1)
        nn_dists_lidar = nn_dists_lidar.reshape(N_down, -1)
        nn_dists_inv_uss = nn_dists_inv_uss.reshape(N_down, -1)
        nn_dists_inv_tof = nn_dists_inv_tof.reshape(N_down, -1)
        nn_dists_inv_nerf = nn_dists_inv_nerf.reshape(N_down, -1)
        nn_dists_inv_lidar = nn_dists_inv_lidar.reshape(N_down, -1)

        if use_relaative_error:
            nn_dists_uss = nn_dists_uss / depth_w_gt_uss
            nn_dists_tof = nn_dists_tof / depth_w_gt_tof
            nn_dists_nerf = nn_dists_nerf / depth_w_gt_nerf
            nn_dists_lidar = nn_dists_lidar / depth_w_gt_lidar
            nn_dists_inv_uss = nn_dists_inv_uss / depth_w_gt_uss
            nn_dists_inv_tof = nn_dists_inv_tof / depth_w_gt_tof
            nn_dists_inv_nerf = nn_dists_inv_nerf / depth_w_gt_nerf
            nn_dists_inv_lidar = nn_dists_inv_lidar / depth_w_gt_lidar

        for i in range(N_down):
            fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(9,10))

            ax = axes[0,0]
            ax.imshow(scan_imgs_uss[i].swapaxes(0,1), origin='lower', extent=extent)
            for j in np.linspace(0, rays_o_w_uss.shape[1]-1, num_ray_steps, dtype=int):
                xs = [rays_o_w_uss[i,j,0], rays_o_w_uss[i,j,0] + depth_w_uss[i,j] * np.cos(scan_angles_uss[i,j])]
                ys = [rays_o_w_uss[i,j,1], rays_o_w_uss[i,j,1] + depth_w_uss[i,j] * np.sin(scan_angles_uss[i,j])]
                ax.plot(xs, ys, c='blue', linewidth=0.1)
            ax.scatter(rays_o_w_uss[i,0,0], rays_o_w_uss[i,0,1], c='red', s=5)
            ax.scatter(rays_o_w_uss[i,-1,0], rays_o_w_uss[i,-1,1], c='red', s=5)
            ax.set_title(f'Scan', fontsize=15, weight='bold')
            ax.set_xlabel(f'x [m]')
            ax.set_ylabel('USS', fontsize=15, weight='bold', labelpad=20)
            ax.text(-0.15, 0.5, 'y [m]', fontsize=10, va='center', rotation='vertical', transform=ax.transAxes)


            ax = axes[1,0]
            ax.imshow(scan_imgs_tof[i].swapaxes(0,1), origin='lower', extent=extent)
            for j in np.linspace(0, rays_o_w_tof.shape[1]-1, num_ray_steps, dtype=int):
                xs = [rays_o_w_tof[i,j,0], rays_o_w_tof[i,j,0] + depth_w_tof[i,j] * np.cos(scan_angles_tof[i,j])]
                ys = [rays_o_w_tof[i,j,1], rays_o_w_tof[i,j,1] + depth_w_tof[i,j] * np.sin(scan_angles_tof[i,j])]
                ax.plot(xs,  ys, c='lime', linewidth=0.1)
            ax.scatter(rays_o_w_tof[i,0,0], rays_o_w_tof[i,0,1], c='red', s=5)
            ax.scatter(rays_o_w_tof[i,-1,0], rays_o_w_tof[i,-1,1], c='red', s=5)
            ax.set_xlabel(f'x [m]')
            ax.set_ylabel('ToF', fontsize=15, weight='bold', labelpad=20)
            ax.text(-0.15, 0.5, 'y [m]', fontsize=10, va='center', rotation='vertical', transform=ax.transAxes)

            ax = axes[2,0]
            ax.imshow(scan_imgs_lidar[i].swapaxes(0,1), origin='lower', extent=extent)
            for j in np.linspace(0, rays_o_w_lidar.shape[1]-1, num_ray_steps, dtype=int):
                xs = [rays_o_w_lidar[i,j,0], rays_o_w_lidar[i,j,0] + depth_w_lidar[i,j] * np.cos(scan_angles_lidar[i,j])]
                ys = [rays_o_w_lidar[i,j,1], rays_o_w_lidar[i,j,1] + depth_w_lidar[i,j] * np.sin(scan_angles_lidar[i,j])]
                ax.plot(xs, ys, c='magenta', linewidth=0.1)
            ax.scatter(rays_o_w_lidar[i,0,0], rays_o_w_lidar[i,0,1], c='red', s=5)
            ax.scatter(rays_o_w_lidar[i,-1,0], rays_o_w_lidar[i,-1,1], c='red', s=5)
            ax.set_xlabel(f'x [m]')
            ax.set_ylabel('Lidar', fontsize=15, weight='bold', labelpad=20)
            ax.text(-0.15, 0.5, 'y [m]', fontsize=10, va='center', rotation='vertical', transform=ax.transAxes)
            
            ax = axes[3,0]
            ax.imshow(scan_imgs_nerf[i].swapaxes(0,1), origin='lower', extent=extent)
            for j in np.linspace(0, rays_o_w_nerf.shape[1]-1, num_ray_steps, dtype=int):
                xs = [rays_o_w_nerf[i,j,0], rays_o_w_nerf[i,j,0] + depth_w_nerf[i,j] * np.cos(scan_angles_nerf[i,j])]
                ys = [rays_o_w_nerf[i,j,1], rays_o_w_nerf[i,j,1] + depth_w_nerf[i,j] * np.sin(scan_angles_nerf[i,j])]
                ax.plot(xs, ys, c='deeppink', linewidth=0.1)
            ax.scatter(rays_o_w_nerf[i,0,0], rays_o_w_nerf[i,0,1], c='red', s=5)
            ax.scatter(rays_o_w_nerf[i,-1,0], rays_o_w_nerf[i,-1,1], c='red', s=5)
            ax.set_xlabel(f'x [m]')
            ax.set_ylabel('NeRF', fontsize=15, weight='bold', labelpad=20)
            ax.text(-0.15, 0.5, 'y [m]', fontsize=10, va='center', rotation='vertical', transform=ax.transAxes)

            ax = axes[0,1]
            val_idxs = ~np.isnan(nn_dists_uss[i])
            n_uss, _, _ = ax.hist(nn_dists_uss[i][val_idxs], bins=hist_bins)
            ax.vlines(np.nanmean(nn_dists_uss[i]), ymin=0, ymax=2000, colors='r', linestyles='dashed', 
                      label=f'MNNE={np.nanmean(nn_dists_uss[i]):.2f}m')
            ax.set_title(f'NNE Sensor->GT', weight='bold')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'NNE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[1,1]
            val_idxs = ~np.isnan(nn_dists_tof[i])
            n_tof, _, _ = ax.hist(nn_dists_tof[i][val_idxs], bins=hist_bins)
            ax.vlines(np.nanmean(nn_dists_tof[i]), ymin=0, ymax=2000, colors='r', linestyles='dashed', 
                      label=f'MNNE={np.nanmean(nn_dists_tof[i]):.2f}m')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'NNE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[2,1]
            val_idxs = ~np.isnan(nn_dists_lidar[i])
            n_lidar, _, _ = ax.hist(nn_dists_lidar[i][val_idxs], bins=hist_bins)
            ax.vlines(np.nanmean(nn_dists_lidar[i]), ymin=0, ymax=2000, colors='r', linestyles='dashed',
                        label=f'MNNE={np.nanmean(nn_dists_lidar[i]):.2f}m')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'NNE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[3,1]
            val_idxs = ~np.isnan(nn_dists_nerf[i])
            n_nerf, _, _ = ax.hist(nn_dists_nerf[i][val_idxs], bins=hist_bins)
            ax.vlines(np.nanmean(nn_dists_nerf[i]), ymin=0, ymax=2000, colors='r', linestyles='dashed', 
                      label=f'MNNE={np.nanmean(nn_dists_nerf[i]):.2f}m')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'NNE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[0,2]
            val_idxs = ~np.isnan(nn_dists_inv_uss[i])
            n_uss_inv, _, _ = ax.hist(nn_dists_inv_uss[i][val_idxs], bins=hist_bins)
            ax.vlines(inlier_thr, ymin=0, ymax=2000, colors='r', linestyles='dashed', 
                      label=f'Inliers={(np.nansum(nn_dists_inv_uss[i]<inlier_thr)/nn_dists_inv_uss.shape[1]):.2f}%')
            ax.set_title(f'NNE GT->Sensor', weight='bold')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'NNE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[1,2]
            val_idxs = ~np.isnan(nn_dists_inv_tof[i])
            n_tof_inv, _, _ = ax.hist(nn_dists_inv_tof[i][val_idxs], bins=hist_bins)
            ax.vlines(inlier_thr, ymin=0, ymax=2000, colors='r', linestyles='dashed', 
                      label=f'Inliers={(np.nansum(nn_dists_inv_tof[i]<inlier_thr)/nn_dists_inv_tof.shape[1]):.2f}%')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'NNE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[2,2]
            val_idxs = ~np.isnan(nn_dists_inv_lidar[i])
            n_lidar_inv, _, _ = ax.hist(nn_dists_inv_lidar[i][val_idxs], bins=hist_bins)
            ax.vlines(inlier_thr, ymin=0, ymax=2000, colors='r', linestyles='dashed',
                        label=f'Inliers={(np.nansum(nn_dists_inv_lidar[i]<inlier_thr)/nn_dists_inv_lidar.shape[1]):.2f}%')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'NNE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[3,2]
            val_idxs = ~np.isnan(nn_dists_inv_nerf[i])
            n_nerf_inv, _, _ = ax.hist(nn_dists_inv_nerf[i][val_idxs], bins=hist_bins)
            ax.vlines(inlier_thr, ymin=0, ymax=2000, colors='r', linestyles='dashed', 
                      label=f'Inliers={(np.nansum(nn_dists_inv_nerf[i]<inlier_thr)/nn_dists_inv_nerf.shape[1]):.2f}%')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'NNE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[0,3]
            if use_relaative_error:
                error = np.abs(depth_w_gt_uss[i] - depth_w_uss[i]) / depth_w_gt_uss[i]
                rmse = np.sqrt(np.nanmean((depth_w_uss[i]**2 - depth_w_gt_uss[i]**2) / depth_w_gt_uss[i]**2))
            else:
                error = np.abs(depth_w_gt_uss[i] - depth_w_uss[i])
                rmse = np.sqrt(np.nanmean(depth_w_uss[i]**2 - depth_w_gt_uss[i]**2))
            val_idxs = (~np.isnan(error))
            n_uss_error, _, _ = ax.hist(error[val_idxs], bins=hist_bins)
            ax.vlines(inlier_thr, ymin=0, ymax=2000, colors='r', linestyles='dashed', 
                      label=f'RMSE={rmse:.2f}m')
            ax.set_title(f'Absolute Error', weight='bold')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'AE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[1,3]
            if use_relaative_error:
                error = np.abs(depth_w_gt_tof[i] - depth_w_tof[i]) / depth_w_gt_tof[i]
                rmse = np.sqrt(np.nanmean((depth_w_tof[i]**2 - depth_w_gt_tof[i]**2) / depth_w_gt_tof[i]**2))
            else:
                error = np.abs(depth_w_gt_tof[i] - depth_w_tof[i])
                rmse = np.sqrt(np.nanmean(depth_w_tof[i]**2 - depth_w_gt_tof[i]**2))
            val_idxs = ~np.isnan(error)
            n_tof_error, _, _ = ax.hist(error[val_idxs], bins=hist_bins)
            ax.vlines(inlier_thr, ymin=0, ymax=2000, colors='r', linestyles='dashed',
                        label=f'RMSE={rmse:.2f}m')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'AE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[2,3]
            if use_relaative_error:
                error = np.abs(depth_w_gt_lidar[i] - depth_w_lidar[i]) / depth_w_gt_lidar[i]
                rmse = np.sqrt(np.nanmean((depth_w_lidar[i]**2 - depth_w_gt_lidar[i]**2) / depth_w_gt_lidar[i]**2))
            else:
                error = np.abs(depth_w_gt_lidar[i] - depth_w_lidar[i])
                rmse = np.sqrt(np.nanmean(depth_w_lidar[i]**2 - depth_w_gt_lidar[i]**2))
            val_idxs = ~np.isnan(error)
            n_lidar_error, _, _ = ax.hist(error[val_idxs], bins=hist_bins)
            ax.vlines(inlier_thr, ymin=0, ymax=2000, colors='r', linestyles='dashed',
                        label=f'RMSE={rmse:.2f}m')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'AE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            ax = axes[3,3]
            if use_relaative_error:
                error = np.abs(depth_w_gt_nerf[i] - depth_w_nerf[i]) / depth_w_gt_nerf[i]
                rmse = np.sqrt(np.nanmean((depth_w_nerf[i]**2 - depth_w_gt_nerf[i]**2) / depth_w_gt_nerf[i]**2))
            else:
                error = np.abs(depth_w_gt_nerf[i] - depth_w_nerf[i])
                rmse = np.sqrt(np.nanmean(depth_w_nerf[i]**2 - depth_w_gt_nerf[i]**2))
            val_idxs = ~np.isnan(error)
            n_nerf_error, _, _ = ax.hist(error[val_idxs], bins=hist_bins)
            ax.vlines(inlier_thr, ymin=0, ymax=2000, colors='r', linestyles='dashed', 
                      label=f'RMSE={rmse:.2f}m')
            ax.set_ylabel(f'# elements')
            ax.set_xlabel(f'AE [m]')
            ax.legend()
            ax.set_box_aspect(1)

            x_max = min(
                max_error_m, 
                np.nanmax(np.concatenate((depth_w_uss[i], depth_w_tof[i], depth_w_nerf[i], depth_w_lidar[i])))
            )
            x_max_inv = min(
                max_error_m, 
                np.nanmax(np.concatenate((nn_dists_inv_uss[i], nn_dists_inv_tof[i], nn_dists_inv_nerf[i], nn_dists_inv_lidar[i])))
            )
            y_max_uss = np.nanmax((n_uss, n_uss_inv, n_uss_error))
            y_max_tof = np.nanmax((n_tof, n_tof_inv, n_tof_error))
            y_max_nerf = np.nanmax((n_nerf, n_nerf_inv, n_nerf_error))
            y_max_lidar = np.nanmax((n_lidar, n_lidar_inv, n_lidar_error))

            axes[0,1].set_xlim([0, x_max])
            axes[0,1].set_ylim([0, y_max_uss])
            axes[1,1].set_xlim([0, x_max])
            axes[1,1].set_ylim([0, y_max_tof])
            axes[2,1].set_xlim([0, x_max])
            axes[2,1].set_ylim([0, y_max_lidar])
            axes[3,1].set_xlim([0, x_max])
            axes[3,1].set_ylim([0, y_max_nerf])
            axes[0,2].set_xlim([0, x_max_inv])
            axes[0,2].set_ylim([0, y_max_uss])
            axes[1,2].set_xlim([0, x_max_inv])
            axes[1,2].set_ylim([0, y_max_tof])
            axes[2,2].set_xlim([0, x_max_inv])
            axes[2,2].set_ylim([0, y_max_lidar])
            axes[3,2].set_xlim([0, x_max_inv])
            axes[3,2].set_ylim([0, y_max_nerf])
            axes[0,3].set_xlim([0, x_max])
            axes[0,3].set_ylim([0, y_max_uss])
            axes[1,3].set_xlim([0, x_max])
            axes[1,3].set_ylim([0, y_max_tof])
            axes[2,3].set_xlim([0, x_max])
            axes[2,3].set_ylim([0, y_max_lidar])
            axes[3,3].set_xlim([0, x_max])
            axes[3,3].set_ylim([0, y_max_nerf])


            for i in range(axes.shape[0]):
                axes[i,0].set_xlim(extent[0], extent[1])
                axes[i,0].set_ylim(extent[2], extent[3])
        
            plt.tight_layout()
            name = f"map{i}_rel" if use_relaative_error else f"map{i}"
            plt.savefig(os.path.join(save_dir, name+".pdf"))
            plt.savefig(os.path.join(save_dir, name+".png"))

    def _plotLosses(
        self,
        logs:dict,
        metrics_dict:dict,
    ):
        """
        Plot losses, mean-nearest-neighbour distance and peak signal-to-noise-ratio.
        Args:
            logs: logs dictionary
            metrics_dict: dict of metrics
        Returns:
            metrics_dict: dict of metrics
        """
        if (not self.args.eval.plot_results) or (self.args.training.max_steps == 0):
            return metrics_dict
        
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,8))

        # plot losses
        ax = axes[0]
        # mask = np.ones(self.args.eval.eval_every_n_steps+1) / (self.args.eval.eval_every_n_steps+1)
        # ax.plot(logs['step'], convolveIgnorNans(logs['loss'], mask), label='total')
        # ax.plot(logs['step'], convolveIgnorNans(logs['color_loss'], mask), label='color')
        # if "rgbd_loss" in logs:
        #     ax.plot(logs['step'], convolveIgnorNans(logs['rgbd_loss'], mask), label='rgbd')
        # if "ToF_loss" in logs:
        #     ax.plot(logs['step'], convolveIgnorNans(logs['ToF_loss'], mask), label='ToF')
        # if "USS_loss" in logs:
        #     # ax.plot(logs['step'], np.convolve(logs['USS_loss'], mask, mode='same'), label='USS')
        #     ax.plot(logs['step'], convolveIgnorNans(logs['USS_close_loss'], mask), label='USS(close)')
        #     ax.plot(logs['step'], convolveIgnorNans(logs['USS_min_loss'], mask), label='USS(min)')
        filter_size = max(self.args.eval.eval_every_n_steps+1, 4)
        ax.plot(logs['step'], smoothIgnoreNans(logs['loss'], filter_size), label='total')
        ax.plot(logs['step'], smoothIgnoreNans(logs['color_loss'], filter_size), label='color')
        if "rgbd_loss" in logs:
            ax.plot(logs['step'], smoothIgnoreNans(logs['rgbd_loss'], filter_size), label='rgbd')
        if "ToF_loss" in logs:
            ax.plot(logs['step'], smoothIgnoreNans(logs['ToF_loss'], filter_size), label='ToF')
        if "USS_loss" in logs:
            # ax.plot(logs['step'], np.convolve(logs['USS_loss'], mask, mode='same'), label='USS')
            ax.plot(logs['step'], smoothIgnoreNans(logs['USS_close_loss'], filter_size), label='USS(close)')
            ax.plot(logs['step'], smoothIgnoreNans(logs['USS_min_loss'], filter_size), label='USS(min)')
        ax.set_ylabel('loss')
        ax.set_ylim([0, 1.0])

        ax.set_xlabel('step')
        secax = ax.secondary_xaxis(
            location='top', 
            functions=(self._step2time, self._time2step),
        )
        secax.set_xlabel('time [s]')
        ax.legend()
        ax.set_title('Losses')

        # plot mnn and psnr 
        if 'mnn' in logs and 'psnr' in logs:
            ax = axes[1]
            color = 'tab:blue'
            not_nan = ~np.isnan(logs['mnn'])
            lns1 = ax.plot(np.array(logs['step'])[not_nan], np.array(logs['mnn'])[not_nan], c=color, label='mnn')
            hln1 = ax.axhline(metrics_dict['NeRF']['mnn'], linestyle="--", c=color, label='mnn final')
            ax.set_ylabel('mnn')
            ax.set_ylim([0, 0.5])
            ax.yaxis.label.set_color('blue') 
            ax.tick_params(axis='y', colors='blue')

            idx1 = dataConverged(
                arr=np.array(logs['mnn'])[not_nan],
                threshold=1.5 * metrics_dict['NeRF']['mnn'],
                data_increasing=False,
            )
            if idx1 != -1:
                vln1 = ax.axvline(np.array(logs['step'])[not_nan][idx1], linestyle=(0, (1, 10)), c="black", label='converged 50%')
                metrics_dict['NeRF']['mnn_converged_50'] = np.array(logs['time'])[not_nan][idx1]
                # print(f"mnn converged 25% at step {logs['step'][idx1]}, idx1={idx1}, threshold={1.25 * metrics_dict['NeRF']['mnn']}")

            idx2 = dataConverged(
                arr=np.array(logs['mnn'])[not_nan],
                threshold=1.25 * metrics_dict['NeRF']['mnn'],
                data_increasing=False,
            )
            if idx2 != -1:
                vln2 = ax.axvline(np.array(logs['step'])[not_nan][idx2], linestyle=(0, (1, 5)), c="black", label='converged 25%')
                metrics_dict['NeRF']['mnn_converged_25'] = np.array(logs['time'])[not_nan][idx2]

            idx3 = dataConverged(
                arr=np.array(logs['mnn'])[not_nan],
                threshold=1.1 * metrics_dict['NeRF']['mnn'],
                data_increasing=False,
            )
            if idx3 != -1:
                vln3 = ax.axvline(np.array(logs['step'])[not_nan][idx3], linestyle=(0, (1, 2)), c="black", label='converged 10%')
                metrics_dict['NeRF']['mnn_converged_10'] = np.array(logs['time'])[not_nan][idx3]

            ax2 = ax.twinx()
            color = 'tab:green'
            not_nan = ~np.isnan(logs['psnr'])
            lns2 = ax2.plot(np.array(logs['step'])[not_nan], np.array(logs['psnr'])[not_nan], label='psnr', c=color)
            # ax.axhline(metrics_dict['NeRF']['psnr'], linestyle="--", c=color, label='psnr final')
            ax2.set_ylabel('psnr')
            ax2.yaxis.label.set_color('green') 
            ax2.tick_params(axis='y', colors='green')

            ax.set_xlabel('step')
            secax = ax.secondary_xaxis(
                location='top', 
                functions=(self._step2time, self._time2step),
            )
            secax.set_xlabel('time [s]')
            lns = lns1 + lns2 + [hln1]
            if idx1 != -1:
                lns += [vln1]
            if idx2 != -1:
                lns += [vln2]
            if idx3 != -1:
                lns += [vln3]
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
            ax.set_title('Metrics')

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "losses.pdf"))
        plt.savefig(os.path.join(self.args.save_dir, "losses.png"))

        return metrics_dict


#  @torch.no_grad()
#     def _plotEvaluation(
#             self,
#             data_w:dict,
#             metrics_dict:dict,
#     ):
#         """
#         Plot scan and density maps.
#         Args:
#             data_w: data dictionary in world coordinates
#             metrics_dict: metrics dictionary
#         """
#         if not self.args.eval.plot_results:
#             return
        
#         M = self.args.eval.res_angular
#         N = data_w['depth'].shape[0] // M
#         if data_w['depth'].shape[0] % M != 0:
#             self.args.logger.error(f"ERROR: trainer_RH.evaluatePlot(): data_w['depth'].shape[0]={data_w['depth'].shape[0]} "
#                                     + f"should be a multiple of M={M}")
        
#         # downsample data
#         depth_w = data_w['depth'].reshape((N, M)) # (N, M)
#         rays_o_w = data_w['rays_o'].reshape((N, M, 3)) # (N, M, 3)
#         scan_angles = data_w['scan_angles'].reshape((N, M)) # (N, M)
#         nn_dists = metrics_dict['nn_dists'] # (N, M)
#         if self.args.eval.num_plot_pts > N:
#             self.args.logger.warning(f"trainer_RH.evaluatePlot(): num_plot_pts={self.args.eval.num_plot_pts} "
#                                         f"should be smaller or equal than N={N}")
#             self.args.eval.num_plot_pts = N
#         elif self.args.eval.num_plot_pts < N:
#             idxs_temp = np.linspace(0, depth_w.shape[0]-1, self.args.eval.num_plot_pts, dtype=int)
#             depth_w = depth_w[idxs_temp]
#             rays_o_w = rays_o_w[idxs_temp]
#             scan_angles = scan_angles[idxs_temp]
#             nn_dists = nn_dists[idxs_temp]  
#         depth_w = depth_w.flatten() # (N*M,)
#         rays_o_w = rays_o_w.reshape((-1, 3)) # (N*M, 3)
#         scan_angles = scan_angles.flatten() # (N*M,)

#         # create scan maps
#         scan_maps = self._scanRays2scanMap(
#             rays_o_w=rays_o_w,
#             depth=depth_w,
#             scan_angles=scan_angles,
#         ) # (N, L, L)
#         scan_map_gt = data_w['scan_map_gt'] # (L, L)

#         # create density maps
#         density_map_gt = self.test_dataset.scene.getSliceMap(
#             height=np.mean(rays_o_w[:,2]), 
#             res=self.args.eval.res_map, 
#             height_tolerance=self.args.eval.height_tolerance, 
#             height_in_world_coord=True
#         ) # (L, L)
#         _, density_map = self.interfereDensityMap(
#             res_map=self.args.eval.res_map,
#             height_w=np.mean(rays_o_w[:,2]),
#             num_avg_heights=self.args.eval.num_avg_heights,
#             tolerance_w=self.args.eval.height_tolerance,
#             threshold=self.args.eval.density_map_thr,
#         ) # (L, L)

#         # create combined maps
#         scan_maps_comb = np.zeros((self.args.eval.num_plot_pts,self.args.eval.res_map,self.args.eval.res_map))
#         for i in range(self.args.eval.num_plot_pts):
#             scan_maps_comb[i] = scan_map_gt + 2*scan_maps[i]
#         density_map_comb = density_map_gt + 2*density_map

#         # plot
#         fig, axes = plt.subplots(ncols=1+self.args.eval.num_plot_pts, nrows=4, figsize=(9,9))

#         scale = self.args.model.scale
#         extent = self.test_dataset.scene.c2w(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
#         extent = extent.T.flatten()

#         ax = axes[0,0]
#         ax.imshow(density_map_gt.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(density_map_comb))
#         ax.set_ylabel(f'GT', weight='bold')
#         ax.set_title(f'Density', weight='bold')
#         ax.set_xlabel(f'x [m]')

#         ax = axes[1,0]
#         ax.imshow(2*density_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(density_map_comb))
#         ax.set_ylabel(f'NeRF', weight='bold')
#         ax.set_xlabel(f'x [m]')

#         ax = axes[2,0]
#         ax.imshow(density_map_comb.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(density_map_comb))
#         ax.set_ylabel(f'GT + NeRF', weight='bold')
#         ax.set_xlabel(f'x [m]')

#         fig.delaxes(axes[3,0])
        
#         rays_o_w = rays_o_w.reshape((-1, self.args.eval.res_angular, 3))
#         for i in range(self.args.eval.num_plot_pts):
#             ax = axes[0,i+1]
#             ax.imshow(scan_map_gt.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(scan_maps_comb[i]))
#             ax.scatter(rays_o_w[i,0,0], rays_o_w[i,0,1], c='w', s=5)
#             ax.scatter(rays_o_w[:,0,0], rays_o_w[:,0,1], c='w', s=5, alpha=0.1)
#             ax.set_title(f'Scan {i+1}', weight='bold')
#             ax.set_xlabel(f'x [m]')
#             ax.set_ylabel(f'y [m]')

#             ax = axes[1,i+1]
#             ax.imshow(2*scan_maps[i].T,origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(scan_maps_comb[i]))
#             # for i in range(0, rays_o_w.shape[0], nb_pts_step):
#             #     for j in range(depth_pos_w.shape[1]):
#             #         ax.plot([rays_o_w[i,j,0], depth_pos_w[i,j,0]], [rays_o_w[i,j,1], depth_pos_w[i,j,1]], c='w', linewidth=0.1)
#             ax.scatter(rays_o_w[i,0,0], rays_o_w[i,0,1], c='w', s=5)
#             ax.scatter(rays_o_w[:,0,0], rays_o_w[:,0,1], c='w', s=5, alpha=0.1)
#             ax.set_xlabel(f'x [m]')
#             ax.set_ylabel(f'y [m]')
            
#             ax = axes[2,i+1]
#             ax.imshow(scan_maps_comb[i].T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(scan_maps_comb[i]))
#             # for i in range(0, rays_o_w.shape[0], nb_pts_step):
#             #     for j in range(depth_pos_w.shape[1]):
#             #         ax.plot([rays_o_w[i,j,0], depth_pos_w[i,j,0]], [rays_o_w[i,j,1], depth_pos_w[i,j,1]], c='w', linewidth=0.1)
#             ax.scatter(rays_o_w[i,0,0], rays_o_w[i,0,1], c='w', s=5)
#             ax.scatter(rays_o_w[:,0,0], rays_o_w[:,0,1], c='w', s=5, alpha=0.1)
#             ax.set_xlabel(f'x [m]')
#             ax.set_ylabel(f'y [m]')

#             ax = axes[3,i+1]
#             ax.hist(nn_dists[i], bins=50)
#             ax.vlines(np.nanmean(nn_dists[i]), ymin=0, ymax=20, colors='r', linestyles='dashed', label=f'avg.={np.nanmean(nn_dists[i]):.2f}')
#             if i == 0:
#                 ax.set_ylabel(f'Nearest Neighbour', weight='bold')
#             else:
#                 ax.set_ylabel(f'# elements')
#             ax.set_xlim([0, np.nanmax(nn_dists)])
#             ax.set_ylim([0, 25])
#             ax.set_xlabel(f'distance [m]')
#             ax.legend()
#             ax.set_box_aspect(1)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.args.save_dir, "maps.pdf"))
#         plt.savefig(os.path.join(self.args.save_dir, "maps.png"))