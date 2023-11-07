import torch
import numpy as np
import matplotlib.pyplot as plt

from args.args import Args
from datasets.robot_at_home_scene import RobotAtHomeScene
from datasets.robot_at_home import RobotAtHome
from datasets.ray_utils import get_rays
from modules.grid import Grid
from modules.rendering import MAX_SAMPLES, render

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
        rh_scene:RobotAtHomeScene=None,
        dataset:RobotAtHome=None,
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
        )


        self.threshold = 0.5
        self.max_init_val = 0.51     

        grid = torch.rand(size=(self.grid_size**3,), device=self.args.device, dtype=torch.float32)
        grid = self.threshold + (self.max_init_val-self.threshold) * grid
        self.occ_3d_grid = grid.reshape(self.grid_size, self.grid_size, self.grid_size)

        # self.occ_3d_grid = 0.505 * torch.ones(grid_size, grid_size, grid_size, device=args.device, dtype=torch.float32)

        self.cell_size = 2*self.args.model.scale / grid_size

        self.I = 32 # number of samples for integral
        self.M = 32 # number of samples for ray measurement

        
        self.decay_warmup = 10
        self.false_detection_prob_every_m = 0.3 # probability of false detection every meter
        max_sensor_range = 25.0 # in meters
        self.std_min = 0.1 # minimum standard deviation of sensor model
        self.std_every_m = 1.0 # standard deviation added every m
        self.attenuation_min = 1.0 # minimum attenuation of sensor model
        self.nerf_correct_prob = 0.02 # probability of nerf interference correctnes

        
        self.prob_min = 0.03 # minimum probability of false detection
        self.attenuation_every_m = 1 / max_sensor_range # attenuation added every m
        self.grid_decay = (0.5/0.51)**(1/self.decay_warmup) # decay of grid probabilities
        self.grid_decay = ((self.grid_decay*1000) // 1) / 1000 # floor to 3 decimals

        if rh_scene is not None:
            self.false_detection_prob_every_m = self.false_detection_prob_every_m / rh_scene.w2cTransformation(pos=1, only_scale=True, copy=False)
            self.std_min = rh_scene.w2cTransformation(pos=self.std_min, only_scale=True, copy=False)
            self.std_every_m = self.std_every_m / rh_scene.w2cTransformation(pos=1, only_scale=True, copy=False)
            self.attenuation_every_m = self.attenuation_every_m / rh_scene.w2cTransformation(pos=1, only_scale=True, copy=False)

        self.update_step = 0





    @torch.no_grad()
    def update(
        self,
        threshold:float,
    ):
        """
        Update grid with image.
        Args:
            threshold: threshold for occupancy grid; float
        Returns:
            grid: occupancy grid; tensor (grid_size, grid_size, grid_size)
        """
        self.update_step += 1

        ray_update, nerf_update = self._sample()

        # print(f"OccGrid: num ray updates: {ray_update['rays_o'].shape[0]}, num nerf updates: {nerf_update['rays_o'].shape[0]}")

        self.rayUpdate(
            rays_o=ray_update["rays_o"],
            rays_d=ray_update["rays_d"],
            meas=ray_update["depth_meas"],
        )
        self.nerfUpdate(
            rays_o=nerf_update["rays_o"],
            rays_d=nerf_update["rays_d"],
            meas=nerf_update["depth_meas"],
            threshold_occ=threshold,
        )

        # if self.update_step > 3:
        #     print("update all cells")
        #     self.nerfUpdateAllCells(
        #         threshold_occ=threshold,
        #     )

        if self.update_step <= self.decay_warmup:
            self.occ_3d_grid *= self.grid_decay

        self.updateBitfield(
            occ_3d=self.occ_3d_grid,
            threshold=threshold,
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
        # get data and calculate rays
        B = self.args.occ_grid.batch_size
        data = self.dataset(
            batch_size=B,
            sampling_strategy=self.args.occ_grid.sampling_strategy,
            origin="occ"
        )

        # choose from which sensor to use the data
        if "RGBD" in data['depth']:
            depth_meas = data['depth']["RGBD"]
        elif "USS" in data['depth'] and "ToF" in data['depth']:
            # depth_meas = torch.cat((data['depth']["USS"][:int(B/2)], data['depth']["ToF"][int(B/2):]), dim=0)
            valid_depth_tof = ~torch.isnan(data['depth']["ToF"])
            depth_meas = data['depth']["ToF"]
            depth_meas[~valid_depth_tof] = data['depth']["USS"][~valid_depth_tof]
        elif "USS" in data['depth']:
            depth_meas = data['depth']["USS"]
        elif "ToF" in data['depth']:
            depth_meas = data['depth']["ToF"]
        else:
            self.args.logger.error("OccupancyGrid.update: no depth sensor found")

        rays_o, rays_d = get_rays(
            directions=data['direction'], 
            c2w=data['pose']
        )           
        self.height_c = torch.mean(rays_o[:, 2]) # TODO: remove

        # choose which updating technique is used: ray update or nerf update
        # ray_update_probs = torch.exp(- (data['sample_count'].to(dtype=torch.float32) - 0.3))
        # if torch.any(ray_update_probs > 1.0) or torch.any(ray_update_probs < 0.0):
        #     print("ERROR: OccupancyGrid.update: ray_update_probs out of range")
        #     torch.clamp(ray_update_probs, 0.0, 1.0)
        # ray_update_true = torch.bernoulli(ray_update_probs).to(dtype=torch.bool)

        ray_update_true = valid_depth_tof

        # remove nan values
        depth_meas_val = ~torch.isnan(depth_meas)
        rays_o = rays_o[depth_meas_val]
        rays_d = rays_d[depth_meas_val]
        depth_meas = depth_meas[depth_meas_val]
        ray_update_true = ray_update_true[depth_meas_val]

        ray_update = {
            "rays_o": rays_o[ray_update_true],
            "rays_d": rays_d[ray_update_true],
            "depth_meas": depth_meas[ray_update_true],
        }
        nerf_update = {
            "rays_o": rays_o[~ray_update_true],
            "rays_d": rays_d[~ray_update_true],
            "depth_meas": depth_meas[~ray_update_true],
        }
        return ray_update, nerf_update



    @torch.no_grad()
    def nerfUpdate(
        self,
        rays_o:torch.Tensor,
        rays_d:torch.Tensor,
        meas:torch.Tensor,
        threshold_occ:float,
    ):
        """
        Update grid by interfering nerf.
        Args:
            rays_o: rays origins; tensor (N, 3)
            rays_d: rays directions; tensor (N, 3)
            meas: measured distance in cube coordinates; tensor (N,)
            threshold_occ: threshold for occupancy grid; float
        """
        # calculate cell distances
        stds = self._calcStd(
            dists=meas,
        ) # (N,)
        steps = torch.linspace(0, 1, self.M, device=self.args.device, dtype=torch.float32) # (M,)
        cell_dists = steps[None,:] * (meas + 5*stds)[:,None] # (N, M)

        # calculate cell positions
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True) # normalize rays
        cell_pos = rays_o[:, None, :] + rays_d[:, None, :] * cell_dists[:, :, None]  # (N, M, 3)

        # add random noise
        noise = torch.rand(size=cell_pos.shape, device=self.args.device, dtype=torch.float32) - 0.5 # (N, M, 3)
        cell_pos = cell_pos + 0.05 * cell_dists[:, :, None] * noise # (N, M, 3)
        cell_pos = cell_pos.reshape(-1, 3) # (N*M, 3)

        cell_density = self.fct_density(
            x=cell_pos,
        ) # (N*M,)
        alpha = - np.log(threshold_occ)
        thrshold_nerf = min(0.01 * MAX_SAMPLES / 3**0.5, torch.mean(cell_density))
        probs_emp = torch.exp(- alpha * cell_density / thrshold_nerf) # (N*M,)
        # probs_emp = self.nerf_correct_prob * probs_emp + (1 - self.nerf_correct_prob) * 0.5 # (N*M,)
        probs_emp = torch.clamp(probs_emp, 0.4, 0.6) # (N*M,)
        probs_occ = 1 - probs_emp # (N*M,)

        # calculate cell indices   
        cell_idxs = self._c2idx(
            pos=cell_pos,
        ) # (N*M, 3)

        # update grid
        self._updateGrid(
            cell_idxs=cell_idxs,
            probs_occ=probs_occ,
            probs_emp=probs_emp,
        )

    @torch.no_grad()
    def rayUpdate(
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
        # calculate probabilities for each cell
        cell_idxs, probs_occ, probs_emp = self._rayProb(
            rays_o=rays_o,
            rays_d=rays_d,
            meas=meas,
        ) # (N*M, 3), (N*M,), (N*M,)

        # update grid
        self._updateGrid(
            cell_idxs=cell_idxs,
            probs_occ=probs_occ,
            probs_emp=probs_emp,
        )

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
        probs = self.occ_3d_grid[cell_idxs[:, 0], cell_idxs[:, 1], cell_idxs[:, 2]] # (N*M,)
        probs = (probs * probs_occ) / (probs * probs_occ + (1 - probs) * probs_emp) # (N*M,)
        self.occ_3d_grid[cell_idxs[:, 0], cell_idxs[:, 1], cell_idxs[:, 2]] = probs

    @torch.no_grad()
    def _rayProb(
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
            meas: measured distance in cube coordinates; tensor (N,)
        Returns:
            cell_idxs: indices of cells to update; tensor (N*M, 3)
            probs_occ: probabilities of measurements given cell is occupied; tensor (N*M,)
            probs_emp: probabilities of measurements given cell is empty; tensor (N*M,)
        """
        # calculate cell distances
        stds = self._calcStd(
            dists=meas,
        ) # (N,)
        steps = torch.linspace(0, 1, self.M, device=self.args.device, dtype=torch.float32) # (M,)
        cell_dists = steps[None,:] * (meas + 5*stds)[:,None] # (N, M)
        

        # calculate cell probabilities
        probs_occ, probs_emp = self._rayMeasProb(
            meas=meas, 
            dists=cell_dists,
        )  # (N, M), (N, M)
        probs_occ = probs_occ.reshape(-1) # (N*M,)
        probs_emp = probs_emp.reshape(-1) # (N*M,)

        # calculate cell indices
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True) # normalize rays
        cell_pos = rays_o[:, None, :] + rays_d[:, None, :] * cell_dists[:, :, None]  # (N, M, 3)
        cell_idxs = self._c2idx(
            pos=cell_pos.reshape(-1, 3),
        ) # (N*M, 3)

        return cell_idxs, probs_occ, probs_emp

    @torch.no_grad()
    def _rayMeasProb(
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
        attenuations = self._calcAttenuation(
            dists=dists,
        )
        stds = self._calcStd(
            dists=dists,
        )
        return attenuations * torch.exp(-0.5 * (meas - dists)**2 / stds**2)
    
    @torch.no_grad()
    def _calcStd(
        self,
        dists:torch.Tensor,
    ):
        """
        Calculate standard deviation of sensor model.
        Args:
            dists: distances to evaluate; tensor (any shape)
        Returns:
            stds: standard deviation of sensor model; tensor (same shape as dists)
        """
        return self.std_min * ( 1 + self.std_every_m*dists)
    
    @torch.no_grad()
    def _calcAttenuation(
        self,
        dists:torch.Tensor,
    ):
        """
        Calculate attenuation of sensor model.
        Args:
            dists: distances to evaluate; tensor (any shape)
        Returns:
            attenuations: attenuation of sensor model; tensor (same shape as dists)
        """
        return self.attenuation_min * (1 - torch.minimum(torch.ones_like(dists, device=self.args.device, dtype=torch.float32), self.attenuation_every_m*dists))

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


    @torch.no_grad()
    def nerfUpdateAllCells(
        self,
        threshold_occ:float,
    ):
        """
        Update grid by interfering nerf.
        Args:
            threshold_occ: threshold for occupancy grid; float
        """
        # define cell indices and positions
        cell_idxs = create_meshgrid3d(
            self.grid_size, 
            self.grid_size, 
            self.grid_size, 
            False, 
            dtype=torch.int32
        ).reshape(-1, 3).to(device=self.args.device) # (N, 3)
        cell_pos = self._idx2c(
            idx=cell_idxs,
        ) # (N, 3)

        # add random noise
        noise = torch.rand(size=cell_pos.shape, device=self.args.device, dtype=torch.float32)
        cell_pos = cell_pos + self.cell_size * noise - self.cell_size/2
        cell_pos = torch.clamp(cell_pos, -self.args.model.scale, self.args.model.scale) # limit to cube

        # calculate cell occupancy probabilities
        cell_density = self.fct_density(
            x=cell_pos,
        ) # (N,)
        alpha = - np.log(threshold_occ)
        thrshold_nerf = 0.01 * MAX_SAMPLES / 3**0.5
        probs_emp = torch.exp(- alpha * cell_density / thrshold_nerf) # (N,)
        probs_occ = 1 - probs_emp # (N,)

        # update grid
        self._updateGrid(
            cell_idxs=cell_idxs,
            probs_occ=probs_occ,
            probs_emp=probs_emp,
        )