import numpy as np
import matplotlib.pyplot as plt

from datasets.ray_utils import get_rays
from training.trainer_rh import TrainerRH
from training.metrics import Metrics


def main():
    res_map = 256
    height_tol = 0.005
    num_imgs = 10
    num_pixs = 100
    alphas = np.linspace(0.8, 1.2, 3)
    betas = np.linspace(0, 0.5, 3)

    hparams = "rh_anto_kitchen_win.json" # "rh_windows.json"
    trainer = TrainerRH(hparams_file=hparams)

    # metric class
    W, H = trainer.test_dataset.img_wh
    metrics = Metrics(
        args=trainer.args,
        img_wh=(W,H)
    )

    # create image and pixel indices
    
    pix_step = W // num_pixs

    img_idxs = np.linspace(0, len(trainer.test_dataset), num_imgs, dtype=int) # (num_imgs,)
    img_idxs = np.repeat(img_idxs, num_pixs) # (num_imgs*num_pixs,)

    pix_idxs = np.arange(W*H).reshape(H, W) # (H, W)
    pix_idxs = pix_idxs[H//2,::pix_step] # (num_pixs,)
    pix_idxs = np.tile(pix_idxs, num_imgs) # (num_imgs*num_pixs,)
    
    # get positions and directions
    direction = trainer.test_dataset.directions[pix_idxs]
    pose = trainer.test_dataset.poses[img_idxs]
    rays_o, rays_d = get_rays(direction, pose)

    scan_map_gt, depth_w_gt, scan_angles = trainer.test_dataset.scene.getSliceScan(
        res=res_map, 
        rays_o=rays_o, 
        rays_d=rays_d, 
        rays_o_in_world_coord=False, 
        height_tolerance=height_tol
    ) # (L, L)

    nn_errors = np.zeros((len(alphas), len(betas)))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # get depths and convert it to world coordinates
            depth_c = trainer.test_dataset.depths[img_idxs, pix_idxs] # (num_imgs*num_pixs,)
            depth_c = depth_c.clone().detach().cpu().numpy()
            depth_c = depth_c*alpha + beta
            depth_w = trainer.test_dataset.scene.c2wTransformation(pos=depth_c, only_scale=True, copy=True)

            # calculate nearest neighbor error
            metric_dict = metrics.evaluate(
                data={
                    "depth_w": depth_w,
                    "depth_w_gt": depth_w_gt,
                },
                eval_metrics=["nn"],
                convert_to_world_coords=False,
                copy=True,
                num_test_pts=num_imgs,
            )
            nn_errors[i,j] = metric_dict["nn"]

    # print best alpha and beta
    best_alpha = alphas[np.argmin(np.mean(nn_errors, axis=1))]
    best_beta = betas[np.argmin(np.mean(nn_errors, axis=0))]
    best_idx = np.argmin(nn_errors)
    best_alpha_idx = best_idx // len(betas)
    best_beta_idx = best_idx % len(betas)
    best_pair = (alphas[best_alpha_idx], betas[best_beta_idx])
    print(f"best alpha: {best_alpha}")
    print(f"best beta: {best_beta}")
    print(f"best pair (alpha,beta): {best_pair}")

    # plot nn errors
    plt.figure()
    plt.imshow(nn_errors)
    plt.colorbar()
    plt.xticks(np.arange(len(betas)), betas)
    plt.yticks(np.arange(len(alphas)), alphas)

    plt.show()
   
if __name__ == "__main__":
    main()