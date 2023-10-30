import torch
import numpy as np

from kornia.utils.grid import create_meshgrid3d
from modules.rendering import NEAR_DISTANCE
from modules.utils import (
    morton3D, 
    morton3D_invert, 
    packbits, 
)
from einops import rearrange

from args.args import Args


class NeRFGrid(torch.nn.Module):
    def __init__(
        self,
        args:Args,
        grid_size:int,
        fct_density:callable,
    ):
        super().__init__()

        self.args = args
        self.grid_size = grid_size
        self.scale = args.model.scale
        self.fct_density = fct_density

        self.cascades = max(1 + int(np.ceil(np.log2(2 * self.scale))), 1)
        
        self.grid = torch.zeros(self.cascades, self.grid_size**3, device=self.args.device)
        self.grid_coords = create_meshgrid3d(
            self.grid_size, 
            self.grid_size, 
            self.grid_size, 
            False, 
            dtype=torch.int32
        ).reshape(-1, 3).to(device=self.args.device)

        # self.register_buffer(
        #     'density_grid',
        #     torch.zeros(self.cascades, self.grid_size**3),

        # self.register_buffer(
        #     'grid_coords',
        #     create_meshgrid3d(
        #         self.grid_size, 
        #         self.grid_size, 
        #         self.grid_size, 
        #         False,
        #         dtype=torch.int32
        #     ).reshape(-1, 3),
        # )

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3),
                                    dtype=torch.int32,
                                    device=self.args.device)
            indices1 = morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(
                self.grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M, ),
                                         device=self.args.device)
                indices2 = indices2[rand_idx]
            coords2 = morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1,
                                  indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=32**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts
        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a')  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ poses[:, :3, 3:]  # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i + chunk] / (self.grid_size - 1) * 2 - 1
                s = min(2**(c - 1), self.scale)
                half_grid_size = s / self.grid_size
                xyzs_w = (xyzs * (s - half_grid_size)).T  # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2] >=
                                  NEAR_DISTANCE) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2] <
                                   NEAR_DISTANCE) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)
                self.grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)
                
    @torch.no_grad()
    def update_density_grid(
        self,
        density_threshold,
        warmup=False,
        decay=0.95,
        erode=False       
    ):
        density_grid_tmp = torch.zeros_like(self.grid)
        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(
                self.grid_size**3 // 4, density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords /
                      (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.fct_density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1 / self.count_grid), 0.1, 0.95)
        self.grid = \
            torch.where(self.grid<0,
                        self.grid,
                        torch.maximum(self.grid*decay, density_grid_tmp))

        return self.grid
