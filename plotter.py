import numpy as np
import matplotlib.pyplot as plt

from trainer_RH import TrainerRH



def plotTrainerRHSlice(trainer:TrainerRH, res:int, heights_w:list, tolerance_w:float, thresholds:list):   
    # Create subplots
    fig, axes = plt.subplots(ncols=2+len(thresholds), nrows=len(heights_w), figsize=(12,6))
    extent_c = np.array([[trainer.train_dataset.scene.w2c_params["cube_min"], trainer.train_dataset.scene.w2c_params["cube_min"]], 
                         [trainer.train_dataset.scene.w2c_params["cube_max"], trainer.train_dataset.scene.w2c_params["cube_max"]]])
    extent_w = trainer.train_dataset.scene.c2wTransformation(pos=extent_c)
    extent = [extent_w[0,0], extent_w[1,0], extent_w[0,1], extent_w[1,1]]
    
    for i, h_w in enumerate(heights_w):
        # estimate density of slice
        density = trainer.evaluateSlice(res=res, height_w=h_w, tolerance_w=tolerance_w) 

        # threshold density   
        density_thresholded = []
        for j, thr in enumerate(thresholds):
            density_thresholded.append(density.copy())
            density_thresholded[j][density < thr] = 0.0
            density_thresholded[j][density >= thr] = 1.0

        # get ground truth
        slice_map = trainer.train_dataset.scene.getSliceMap(height=heights_w[i], res=res, height_tolerance=tolerance_w, height_in_world_coord=True)

        # plot the ground truth
        ax = axes[i,0]
        ax.imshow(slice_map, extent=extent, origin='lower', cmap='viridis')
        if i == 0:
            ax.set_title(f'Ground Truth')
        ax.set_ylabel(f'Height {heights_w[i]}m')

        # Plot the density map for the current subplot
        ax = axes[i,1]
        ax.imshow(density, extent=extent, origin='lower', cmap='viridis')
        if i == 0:
            ax.set_title(f'Rendered Density')

        for j, sig_thr in enumerate(density_thresholded):
            ax = axes[i, j+2]
            ax.imshow(sig_thr, extent=extent, origin='lower', cmap='viridis')
            if i == 0:
                ax.set_title(f'Threshold = {thresholds[j]}')

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    plt.show()

def plotTrainerRHScan(trainer:TrainerRH, res:int, res_angular:int, np_test_pts:int=3):

    depth_mse, depth_w, depth_w_gt, slice_map_gt, rays_o_c, scan_angles = trainer.evaluateDepth(res=res, res_angular=res_angular)


    # convert scan depth to position in world coordinate system
    rays_o_w = trainer.test_dataset.scene.c2wTransformation(pos=rays_o_c, copy=True)
    depth_pos_w = trainer.test_dataset.scene.convertDepth2Pos(rays_o=rays_o_w, scan_depth=depth_w, scan_angles=scan_angles) # (N*M, 2)

    # create slice map
    slice_map = np.zeros((res,res))
    slice_map_idxs = trainer.test_dataset.scene.w2idxTransformation(pos=depth_pos_w, res=res) # (N*M, 2)
    slice_map[slice_map_idxs[:,0], slice_map_idxs[:,1]] = 1.0

    # plot
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,6))
    extent = trainer.test_dataset.scene.c2wTransformation(pos=np.array([[-0.5,-0.5],[0.5,0.5]]), copy=False)
    extent = extent.T.flatten()
    comb_map = slice_map_gt + 2*slice_map
    # score = np.sum(slice_map * slice_scans[i]) / np.sum(slice_scans[i])

    # reshape
    rays_o_w = rays_o_w.reshape((-1, res_angular, 3))
    depth_pos_w = depth_pos_w.reshape((-1, res_angular, 2))

    nb_pts_step = rays_o_w.shape[0]//np_test_pts
    nb_pts_step = 1 if nb_pts_step == 0 else nb_pts_step

    ax = axes[0]
    ax.imshow(slice_map_gt.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
    for i in range(0, rays_o_w.shape[0], nb_pts_step):
        ax.scatter(rays_o_w[i,0,0], rays_o_w[i,0,1], c='w', s=5)
    ax.set_title(f'Map')
    ax.set_xlabel(f'x [m]')
    ax.set_ylabel(f'y [m]')

    ax = axes[1]
    ax.imshow(2*slice_map.T,origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
    for i in range(0, rays_o_w.shape[0], nb_pts_step):
        for j in range(depth_pos_w.shape[1]):
            ax.plot([rays_o_w[i,j,0], depth_pos_w[i,j,0]], [rays_o_w[i,j,1], depth_pos_w[i,j,1]], c='w', linewidth=0.1)
        ax.scatter(rays_o_w[i,0,0], rays_o_w[i,0,1], c='w', s=5)
    ax.set_title(f'Scan')
    ax.set_xlabel(f'x [m]')
    
    ax = axes[2]
    ax.imshow(comb_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
    for i in range(0, rays_o_w.shape[0], nb_pts_step):
        for j in range(depth_pos_w.shape[1]):
            ax.plot([rays_o_w[i,j,0], depth_pos_w[i,j,0]], [rays_o_w[i,j,1], depth_pos_w[i,j,1]], c='w', linewidth=0.1)
        ax.scatter(rays_o_w[i,0,0], rays_o_w[i,0,1], c='w', s=5)
    ax.set_title(f'Combined')
    ax.set_xlabel(f'x [m]')
    
    plt.tight_layout()
    plt.show()  




def test_plotTrainerRHSlice():
    ckpt_path = "results/rh_anto_livingroom1/model.pth"
    trainer = TrainerRH()
    trainer.loadCheckpoint(ckpt_path=ckpt_path)

    # create slice
    res = 128
    heights_w = [0.7045, 1.045, 1.345] # in scene coordinates (meters)
    tolerance_w = 0.1 # in meters
    thresholds = [5, 10, 15, 20]

    plotTrainerRHSlice(trainer=trainer, res=res, heights_w=heights_w, tolerance_w=tolerance_w, thresholds=thresholds)


def test_plotTrainerRHSlice():
    ckpt_path = "results/rh_anto_livingroom1/model.pth"
    trainer = TrainerRH()
    trainer.loadCheckpoint(ckpt_path=ckpt_path)

    # create slice
    res = 128
    res_angular = 256
    np_test_pts = 3
    plotTrainerRHScan(trainer=trainer, res=res, res_angular=res_angular, np_test_pts=np_test_pts)


if __name__ == '__main__':
    # test_plotTrainerRHSlice()
    test_plotTrainerRHSlice()