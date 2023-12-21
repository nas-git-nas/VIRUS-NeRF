import torch
import numpy as np
import matplotlib.pyplot as plt

from args.args import Args
from datasets.scene_base import SceneBase
from datasets.dataset_base import DatasetBase
from datasets.ray_utils import get_rays
from modules.grid import Grid
from modules.rendering import MAX_SAMPLES, render
from helpers.geometric_fcts import distToCubeBorder

from kornia.utils.grid import create_meshgrid3d

from modules.rendering import NEAR_DISTANCE
from modules.utils import (
    morton3D, 
    morton3D_invert, 
    packbits, 
)


class OccupancyGrid(Grid):
    def __init__(
        self,
        args:Args,
        grid_size:int,
        scene:SceneBase=None,
        dataset:DatasetBase=None,
        fct_density:callable=None,
    ) -> None:
        
        self.args = args
        self.grid_size = grid_size
        self.dataset = dataset
        self.fct_density = fct_density
        self.cascades = max(1 + int(np.ceil(np.log2(2 * self.args.model.scale))), 1)

        super().__init__(
            args=args,
            grid_size=grid_size,
            cascades=self.cascades,
            morton_structure=False,
        )

        self.update_step = 0 # update counter

        # initialize occupancy grid
        occ_threshold = 0.5
        occ_init_max = 0.51     
        grid = torch.rand(size=(self.grid_size**3,), device=self.args.device, dtype=torch.float32)
        grid = occ_threshold + (occ_init_max-occ_threshold) * grid
        self.occ_3d_grid = grid.reshape(self.grid_size, self.grid_size, self.grid_size)
        
        # fixed parameters
        self.I = 32 # number of samples for integral
        self.M = 32 # number of samples for ray measurement
        self.prob_min = 0.03 # minimum probability of false detection

        # variable parameters
        decay_num_steps = self.args.occ_grid.decay_warmup_steps / self.args.occ_grid.update_interval
        grid_decay = (occ_threshold/occ_init_max)**(1/decay_num_steps) # decay of grid probabilities
        self.grid_decay = ((grid_decay*1000) // 1) / 1000 # floor to 3 decimals
        self.cell_size = 2*self.args.model.scale / grid_size

        if scene is not None: # TODO: remove this
            self.false_detection_prob_every_m = self.args.occ_grid.false_detection_prob_every_m / scene.w2c(pos=1, only_scale=True, copy=False)
            self.std_every_m = scene.w2c(pos=self.args.occ_grid.std_every_m, only_scale=True, copy=False)
        else:
            self.false_detection_prob_every_m = self.args.occ_grid.false_detection_prob_every_m
            self.std_every_m = self.args.occ_grid.std_every_m

    @torch.no_grad()
    def update(
        self,
        threshold:float,
    ):
        """
        Update grid with image.
        Args:
            threshold: threshold for occupancy grid; float # TODO: remove this
        Returns:
            grid: occupancy grid; tensor (grid_size, grid_size, grid_size)
        """
        # sample depth measurements
        ray_update, nerf_update = self._sample()

        # update occupancy grid
        if ray_update["batch_size"] > 0:
            self._rayUpdate(
                rays_o=ray_update["rays_o"],
                rays_d=ray_update["rays_d"],
                meas=ray_update["depth_meas"],
            )
        if nerf_update["batch_size"] > 0:
            self._nerfUpdate(
                rays_o=nerf_update["rays_o"],
                rays_d=nerf_update["rays_d"],
                meas=nerf_update["depth_meas"],
            )

        # warmup decay
        self.update_step += 1
        if self.update_step <= self.args.occ_grid.decay_warmup_steps:
            self.occ_3d_grid *= self.grid_decay
        
        # update binary bitfield
        self.threshold = threshold
        self.updateBitfield(
            grid=self.occ_3d_grid,
            threshold=threshold,
            convert_cart2morton=True,
        ) 

    @torch.no_grad()
    def _sample(
        self,
    ):
        """
        Sample data, choose sensor for depth measurement and choose updating technique.
        Returns:
            ray_update: data for ray update; dict
            nerf_update: data for nerf update; dict
        """
        B = self.args.occ_grid.batch_size
        B_ray = int(B * self.args.occ_grid.batch_ratio_ray_update)
        B_nerf = B - B_ray  
        
        # sample data for ray and nerf updates 
        if self.args.occ_grid.sampling_strategy['rays'] == "rgbd":
            ray_update = self._sampleBatch(
                B=B_ray,
                sensor="RGBD",
            )
            nerf_update = self._sampleBatch(
                B=B_nerf,
                sensor="RGBD",
            )
        elif self.args.occ_grid.sampling_strategy['rays'] == "uss_tof": 
            ray_update = self._sampleBatch(
                B=B_ray,
                sensor="ToF",
            )
            nerf_update = self._sampleBatch(
                B=B_nerf,
                sensor="USS",
            )
        else:
            self.args.logger.error("occupancy grid sampling strategy does not exist")

        return ray_update, nerf_update

    @torch.no_grad()
    def _sampleBatch(
        self,
        B:int,
        sensor:str,
    ):
        """
        Sample a batch of data from particular sensor.
        Args:
            B: batch size; int
            sensor: sensor name; str
        Returns:
            update_dict: measurements; dict
                'batch_size: batch size; int
                'rays_o': ray origins; tensor of floats (B, 3)
                'rays_d': ray directions; tensor of floats (B, 3)
                'depth_meas': depth measurements; tensor of floats (B,)
        """
        data = self.dataset(
            batch_size=B,
            sampling_strategy={ "imgs": self.args.occ_grid.sampling_strategy['imgs'], "rays": "valid_"+sensor.lower() },
            origin="occ"
        )

        rays_o = data['rays_o']
        rays_d = data['rays_d']
        depth_meas = data['depth'][sensor]
        valid_depth = ~torch.isnan(depth_meas)
        return {
            "batch_size": B,
            "rays_o": rays_o[valid_depth],
            "rays_d": rays_d[valid_depth],
            "depth_meas": depth_meas[valid_depth],
        }   

    @torch.no_grad()
    def _rayUpdate(
        self,
        rays_o:torch.Tensor,
        rays_d:torch.Tensor,
        meas:torch.Tensor,
    ):
        """
        Update grid with ray measurement.
        Args:
            rays_o: rays origins; tensor (N, 3)
            rays_d: rays directions; tensor (N, 3)
            meas: measured distance in cube coordinates; tensor (N, 1)
        """
        # calculate cell distances and indices
        cell_dists, _, cell_idxs = self._calcPos(
            rays_o=rays_o,
            rays_d=rays_d,
            meas=meas,
            add_noise=False,
        ) # (N, M), _, (N*M, 3)  

        # calculate cell probabilities
        probs_occ, probs_emp = self._rayProb(
            meas=meas, 
            dists=cell_dists,
        )  # (N, M), (N, M)
        probs_occ = probs_occ.reshape(-1) # (N*M,)
        probs_emp = probs_emp.reshape(-1) # (N*M,)

        # update grid
        self._updateGrid(
            cell_idxs=cell_idxs,
            probs_occ=probs_occ,
            probs_emp=probs_emp,
        )

    @torch.no_grad()
    def _nerfUpdate(
        self,
        rays_o:torch.Tensor,
        rays_d:torch.Tensor,
        meas:torch.Tensor,
    ):
        """
        Update grid by interfering nerf.
        Args:
            rays_o: rays origins; tensor (N, 3)
            rays_d: rays directions; tensor (N, 3)
            meas: measured distance in cube coordinates; tensor (N,)
        """
        # calculate cell positions and indices
        _, cell_pos, cell_idxs = self._calcPos(
            rays_o=rays_o,
            rays_d=rays_d,
            meas=meas,
            add_noise=True,
        ) # _, (N*M, 3), (N*M, 3)

        probs_occ, probs_emp = self._nerfProb(
            cell_pos=cell_pos,
        )

        # update grid
        self._updateGrid(
            cell_idxs=cell_idxs,
            probs_occ=probs_occ,
            probs_emp=probs_emp,
        )
    
    @torch.no_grad()
    def _calcPos(
        self,
        rays_o:torch.Tensor,
        rays_d:torch.Tensor,
        meas:torch.Tensor, # TODO: remove this
        add_noise:bool,
    ):
        """
        Update grid with ray measurement.
        Args:
            rays_o: rays origins; tensor (N, 3)
            rays_d: rays directions; tensor (N, 3)
            meas: measured distance in cube coordinates; tensor (N,)
            add_noise: whether to add noise to cell positions; bool
        Returns:
            cell_dists: distances to cell; tensor (N, M)
            cell_pos: positions of cell; tensor (N*M, 3)
            cell_idxs: indices of cell; tensor (N*M, 3)
        """
        # # calculate cell distances
        # stds = self._calcStd(
        #     dists=meas,
        # ) # (N,)
        # steps = torch.linspace(0, 1, self.M, device=self.args.device, dtype=torch.float32) # (M,)
        # cell_dists = steps[None,:] * (meas + 5*stds)[:,None] # (N, M)

        # calculate cell distances
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True) # normalize rays
        dists_to_border = distToCubeBorder(
            rays_o=rays_o,
            rays_d=rays_d,
            cube_max=self.args.model.scale,
            cube_min=-self.args.model.scale,
        ) # (N,)
        steps = torch.linspace(0, 1, self.M, device=self.args.device, dtype=torch.float32) # (M,)
        cell_dists = steps[None,:] * dists_to_border[:,None] # (N, M)

        # calculate cell positions
        cell_pos = rays_o[:, None, :] + rays_d[:, None, :] * cell_dists[:, :, None]  # (N, M, 3)

        # add random noise
        if add_noise:
            noise = 2*torch.rand(size=cell_pos.shape, device=self.args.device, dtype=torch.float32) - 1 # (N, M, 3)
            cell_pos = cell_pos + self.std_every_m * cell_dists[:, :, None] * noise # (N, M, 3)

        # calculate cell indices
        cell_pos = cell_pos.reshape(-1, 3) # (N*M, 3)
        cell_idxs = self._c2idx(
            pos=cell_pos,
        ) # (N*M, 3)

        return cell_dists, cell_pos, cell_idxs

    @torch.no_grad()
    def _rayProb(
        self,
        meas:torch.Tensor,
        dists:torch.Tensor,
        return_probs:bool=False,
    ):
        """
        Compute probability of measurement given cell is occupied or empty
        P[meas@dist | cell=occ] and P[meas@dist | cell=emp] where
        P[meas@dist | cell] = P[meas=dist | cell] * P[meas not< dist | cell]
            -> P[meas=dist | cell=occ]: sensor model e.g. accuracy of a 
                measurement at a given distance
            -> P[meas=dist | cell=emp]: probability of false detection
            -> P[meas not< dist | cell]: probability of not detecting any 
        Args:
            meas: measured distance in cube coordinates; tensor (N,)
            dists: distances to evaluate; tensor (N, M)
            return_probs: whether to return all probabilities; bool
        Returns:
            probs_occ: probabilities of measurements given cell is occupied; tensor (N, M)
            probs_emp: probabilities of measurements given cell is empty; tensor (N, M)
        """
        # calculating P[meas=dist | cell=emp] and P[meas=dist | cell=occ]
        probs_equal_emp = self._sensorEmptyPDF(
            shape=dists.shape,
        ) # (N, M)
        probs_equal_occ = probs_equal_emp + self._sensorOccupiedPDF(
            meas=meas[:, None],
            dists=dists,
        ) # (N, M)

        # calculating P[meas not< dist | cell=emp]
        probs_notless_emp = 1 - probs_equal_emp * dists # (N, M)
        probs_notless_emp[probs_notless_emp < self.prob_min] = self.prob_min

        # calculating P[meas not< dist | cell=occ]
        y = torch.linspace(0, 1, self.I, device=self.args.device, dtype=torch.float32)[None, :] * meas[:, None] # (N, I)
        integral = self._sensorOccupiedPDF(
            meas=y[:, None, :],
            dists=dists[:, :, None],
        )
        integral = torch.sum(integral, dim=2) * (meas/self.I)[:, None] # (N, M)
        probs_notless_occ = probs_notless_emp - integral # (N, M)
        probs_notless_occ[probs_notless_occ < self.prob_min] = self.prob_min

        # calculating P[meas@dist | cell=emp] and P[meas@dist | cell=occ]
        probs_emp = probs_equal_emp * probs_notless_emp
        probs_occ = probs_equal_occ * probs_notless_occ

        if return_probs:
            return probs_occ, probs_emp, probs_equal_emp, probs_equal_occ, probs_notless_emp, probs_notless_occ
        return probs_occ, probs_emp
    
    @torch.no_grad()
    def _nerfProb(
        self,
        cell_pos:torch.Tensor,
    ):
        # interfere NeRF
        cell_density = self.fct_density(
            x=cell_pos,
        ) # (N*M,)

        # convert density to probability
        threshold_nerf = min(self.args.occ_grid.nerf_threshold_max, torch.mean(cell_density).item())
        h_thr = - np.log(threshold_nerf)
        h = torch.log(cell_density)
        probs_occ = 1 / (1 + torch.exp(- self.args.occ_grid.nerf_threshold_slope * (h - h_thr))) # TODO: optimize
        probs_emp = 1 - probs_occ

        # print(f"_nerfProb: threshold_nerf={threshold_nerf:.3f}; probs occ mean={torch.mean(probs_occ):.3f}," \
        #       f"min={torch.min(probs_occ):.3f}, max={torch.max(probs_occ):.3f}, " \
        #       f"mean_above={torch.mean(probs_occ[probs_occ>0.5]):.3f}, mean_below={torch.mean(probs_occ[probs_occ<0.5]):.3f}")
        
        return probs_occ, probs_emp
    
    @torch.no_grad()
    def _updateGrid(
        self,
        cell_idxs:torch.Tensor,
        probs_occ:torch.Tensor,
        probs_emp:torch.Tensor,
    ):
        """
        Update grid with probabilities.
        Args:
            cell_idxs: indices of cells to update; tensor (N*M, 3)
            probs_occ: probabilities of measurements given cell is occupied; tensor (N*M,)
            probs_emp: probabilities of measurements given cell is empty; tensor (N*M,)
        """
        if self.args.model.debug_mode:
            if torch.any(torch.isnan(probs_occ)) or torch.any(torch.isnan(probs_emp)):
                self.args.logger.warning("NaN values in probabilities")

        probs = self.occ_3d_grid[cell_idxs[:, 0], cell_idxs[:, 1], cell_idxs[:, 2]] # (N*M,)
        probs = (probs * probs_occ) / (probs * probs_occ + (1 - probs) * probs_emp) # (N*M,)
        self.occ_3d_grid[cell_idxs[:, 0], cell_idxs[:, 1], cell_idxs[:, 2]] = probs
    
    @torch.no_grad()
    def _sensorEmptyPDF(
        self,
        shape:tuple,
    ):
        """
        Calculate empty probability density function of sensor model:
        Probabilty that measurement equals to distance given
        that the cell is empty: P[meas=dist | cell=emp].
        Args:
            shape: shape of tensor to return; tuple
        Returns:
            probs: probabilities of measurements given cell is empty; tensor (shape)
        """
        return self.false_detection_prob_every_m * torch.ones(shape, device=self.args.device, dtype=torch.float32)
    
    @torch.no_grad()
    def _sensorOccupiedPDF(
        self,
        meas:torch.Tensor,
        dists:torch.Tensor,
    ):
        """
        Calculate occupied probability density function of sensor model:
        Probabilty that measurement equals to distance given
        that the cell is occupied: P[meas=dist | cell=occ].
        Args:
            meas: measured distance in cube coordinates; tensor (any shape)
            dists: distances to evaluate; tensor (same shape as meas)
        Returns:
            probs: probabilities of measurements given cell is occupied; tensor (same shape as meas)
        """
        stds = self.std_every_m * dists + 0.00001 # avoid division by zero
        return torch.exp(-0.5 * (meas - dists)**2 / stds**2)

    @torch.no_grad()
    def _c2idx(
        self,
        pos:torch.Tensor,
    ):
        """
        Convert cube coordinates to occupancy grid indices.
        Args:
            pos: position in cube coordinates; tensor (N, 2/3)
        Returns:
            idx: occupancy grid indices; tensor (N, 2/3)
        """
        map_idxs = (self.grid_size - 1) * (pos + self.args.model.scale) / (2 * self.args.model.scale) # (N, x y z)
        return torch.clamp(map_idxs.round().to(dtype=torch.int32), 0, self.grid_size-1) # convert to int
    
    @torch.no_grad()
    def _idx2c(
        self,
        idx:torch.Tensor,
    ):
        """
        Convert occupancy grid indices to cube coordinates.
        Args:
            idx: occupancy grid indices; tensor (N, 2/3)
        Returns:
            pos: position in cube coordinates; tensor (N, 2/3)
        """
        pos = (2 * self.args.model.scale) * (idx + 0.5) / self.grid_size - self.args.model.scale
        return torch.clamp(pos, -self.args.model.scale, self.args.model.scale) # limit to cube


    # @torch.no_grad()
    # def nerfUpdateAllCells(
    #     self,
    #     threshold_occ:float,
    # ):
    #     """
    #     Update grid by interfering nerf.
    #     Args:
    #         threshold_occ: threshold for occupancy grid; float
    #     """
    #     # define cell indices and positions
    #     cell_idxs = create_meshgrid3d(
    #         self.grid_size, 
    #         self.grid_size, 
    #         self.grid_size, 
    #         False, 
    #         dtype=torch.int32
    #     ).reshape(-1, 3).to(device=self.args.device) # (N, 3)
    #     cell_pos = self._idx2c(
    #         idx=cell_idxs,
    #     ) # (N, 3)

    #     # add random noise
    #     noise = torch.rand(size=cell_pos.shape, device=self.args.device, dtype=torch.float32)
    #     cell_pos = cell_pos + self.cell_size * noise - self.cell_size/2
    #     cell_pos = torch.clamp(cell_pos, -self.args.model.scale, self.args.model.scale) # limit to cube

    #     # calculate cell occupancy probabilities
    #     cell_density = self.fct_density(
    #         x=cell_pos,
    #     ) # (N,)
    #     alpha = - np.log(threshold_occ)
    #     thrshold_nerf = 0.01 * MAX_SAMPLES / 3**0.5
    #     probs_emp = torch.exp(- alpha * cell_density / thrshold_nerf) # (N,)
    #     probs_occ = 1 - probs_emp # (N,)

    #     # update grid
    #     self._updateGrid(
    #         cell_idxs=cell_idxs,
    #         probs_occ=probs_occ,
    #         probs_emp=probs_emp,
    #     )


    # @torch.no_grad()
    # def _nerfProb(
    #     self,
    #     cell_pos:torch.Tensor,
    #     threshold_occ:float, # TODO: remove this
    # ):
    #     # interfere NeRF
    #     cell_density = self.fct_density(
    #         x=cell_pos,
    #     ) # (N*M,)

    #     # convert density to probability
    #     # alpha = - np.log(threshold_occ)
    #     # threshold_nerf = min(self.nerf_threshold_max, torch.mean(cell_density))
    #     # probs_emp = torch.exp(- alpha * cell_density / threshold_nerf) # (N*M,)
    #     # probs_emp = torch.clamp(probs_emp, 1-self.nerf_prob_max, self.nerf_prob_max) # (N*M,)
    #     # probs_occ = 1 - probs_emp # (N*M,)

    #     # # convert density to probability
    #     # threshold_nerf = min(self.nerf_threshold_max, torch.mean(cell_density))
    #     # delta = max(
    #     #     abs(torch.max(cell_density).item() - threshold_nerf),
    #     #     abs(torch.min(cell_density).item() - threshold_nerf),
    #     # )
    #     # delta = min(delta, 100 * self.nerf_threshold_max) # avoid delta to be infinity
    #     # probs_occ = 0.5 * torch.ones_like(cell_density, device=self.args.device, dtype=torch.float32) # (N*M,)
    #     # probs_occ += (cell_density - threshold_nerf) / (2 * delta) # (N*M,)
    #     # probs_occ = torch.clamp(probs_occ, 0, 1) # (N*M,)
    #     # probs_emp = 1 - probs_occ # (N*M,)

    #     # # convert density to probability
    #     # threshold_nerf = min(self.nerf_threshold_max, torch.mean(cell_density).item())
    #     # h_thr = - np.log(threshold_nerf + 1) / np.log(0.5)
    #     # h = torch.log(cell_density + 1)
    #     # probs_emp = torch.exp(- h / h_thr)
    #     # probs_occ = 1 - probs_emp


    #     @torch.no_grad()
    # def _sensorOccupiedPDF(
    #     self,
    #     meas:torch.Tensor,
    #     dists:torch.Tensor,
    # ):
    #     """
    #     Calculate occupied probability density function of sensor model:
    #     Probabilty that measurement equals to distance given
    #     that the cell is occupied: P[meas=dist | cell=occ].
    #     Args:
    #         meas: measured distance in cube coordinates; tensor (any shape)
    #         dists: distances to evaluate; tensor (same shape as meas)
    #     Returns:
    #         probs: probabilities of measurements given cell is occupied; tensor (same shape as meas)
    #     """
    #     attenuations = self._calcAttenuation(
    #         dists=dists,
    #     )
    #     stds = self._calcStd(
    #         dists=dists,
    #     )
    #     return attenuations * torch.exp(-0.5 * (meas - dists)**2 / stds**2)
    
    # @torch.no_grad()
    # def _calcStd(
    #     self,
    #     dists:torch.Tensor,
    # ):
    #     """
    #     Calculate standard deviation of sensor model.
    #     Args:
    #         dists: distances to evaluate; tensor (any shape)
    #     Returns:
    #         stds: standard deviation of sensor model; tensor (same shape as dists)
    #     """
    #     # return self.std_min * ( 1 + self.std_every_m*dists)
    #     return self.std_every_m * dists
    
    # @torch.no_grad()
    # def _calcAttenuation(
    #     self,
    #     dists:torch.Tensor,
    # ):
    #     """
    #     Calculate attenuation of sensor model.
    #     Args:
    #         dists: distances to evaluate; tensor (any shape)
    #     Returns:
    #         attenuations: attenuation of sensor model; tensor (same shape as dists)
    #     """
    #     attenuation = self.attenuation_min * (1 - self.attenuation_every_m * dists)
        # return torch.clamp(attenuation, 0, 1)































