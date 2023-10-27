import torch
import numpy as np
import matplotlib.pyplot as plt

from args.args import Args
from datasets.robot_at_home_scene import RobotAtHomeScene

class OccupancyGrid():
    def __init__(
        self,
        args:Args,
        grid_size:int,
        rh_scene:RobotAtHomeScene=None,
    ) -> None:
        self.args = args
        self.grid_size = grid_size

        self.grid = 0.5 * torch.ones(grid_size, grid_size, grid_size, device=args.device, dtype=torch.float32)

        self.cell_size = 2*self.args.model.scale / grid_size

        self.I = 32 # number of samples for integral
        self.M = 32 # number of samples for ray measurement

        max_sensor_range = 25.0 # in meters
        self.std_min = 0.2 # minimum standard deviation of sensor model
        self.std_every_m = 1.0 # standard deviation added every m
        self.attenuation_min = 1.0 # minimum attenuation of sensor model
        self.attenuation_every_m = 1 / max_sensor_range # attenuation added every m

        if rh_scene is not None:
            self.std_min = rh_scene.w2cTransformation(pos=self.std_min, only_scale=True, copy=False)
            self.std_every_m = self.std_every_m / rh_scene.w2cTransformation(pos=1, only_scale=True, copy=False)
            self.attenuation_every_m = self.attenuation_every_m / rh_scene.w2cTransformation(pos=1, only_scale=True, copy=False)

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

        return self.grid

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
        probs = self.grid[cell_idxs[:, 0], cell_idxs[:, 1], cell_idxs[:, 2]] # (N*M,)
        probs = (probs * probs_occ) / (probs * probs_occ + (1 - probs) * probs_emp) # (N*M,)
        self.grid[cell_idxs[:, 0], cell_idxs[:, 1], cell_idxs[:, 2]] = probs

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
        cell_dists = steps[None,:] * (meas + 3*stds)[:,None] # (N, M)
        

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

        # calculating P[meas not< dist | cell=emp] and P[meas not< dist | cell=occ]
        probs_notless_emp = 1 - probs_equal_emp * dists # (N, M)
        y = torch.linspace(0, 1, self.I, device=self.args.device, dtype=torch.float32)[None, :] * meas[:, None] # (N, I)
        integral = self._sensorOccupiedPDF(
            meas=y[:, None, :],
            dists=dists[:, :, None],
        )
        integral = torch.sum(integral, dim=2) * (meas/self.I)[:, None] # (N, M)
        probs_notless_occ = probs_notless_emp - integral # (N, M)

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
        return 0.2 * torch.ones(shape, device=self.args.device, dtype=torch.float32)
    
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


