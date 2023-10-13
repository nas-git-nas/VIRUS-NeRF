import numpy as np
import torch
from abc import ABC, abstractmethod
from einops import rearrange

from torchmetrics import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
)

from datasets.robot_at_home_scene import RobotAtHomeScene
from args.args import Args

class Metrics():
    def __init__(
            self,
            args:Args,
            img_wh:tuple,
        ) -> None:
        """
        Metrics base class.
        Args:
            args: arguments; Args object
            img_wh: image width and height; tuple of int (2,)
        """
        self.args = args
        self.img_wh = img_wh

        self.val_psnr = PeakSignalNoiseRatio(
            data_range=1
        ).to(self.args.device)
        self.val_ssim = StructuralSimilarityIndexMeasure(
            data_range=1
        ).to(self.args.device)

    @abstractmethod
    def convertData(self):
        pass

    def evaluate(
            self, 
            data:dict,
            eval_metrics:list,
            convert_to_world_coords:bool=True, 
            copy:bool=True
        ) -> dict:
        """
        Evaluate metrics listed in eval_metrics.
        Args:
            data: data dictionary, containing one or multiple of the following keys:
                depth: predicted depth; either numpy array or torch tensor (N*M,)
                depth_gt: ground truth depth; either numpy array or torch tensor (N*M,)
                rays_o: origin of rays in world coordinates; numpy array (N*M, 3)
                scan_angles: scan angles in radians; numpy array (M,)
                rgb: predicted rgb; either numpy array or torch tensor (N*H*W, 3)
                rgb_gt: ground truth rgb; either numpy array or torch tensor (N*H*W, 3)
            eval_metrics: list of metrics to evaluate; list of str
            convert_to_world_coords: convert depth to world coordinates (meters); bool
            copy: whether or not to copy input arrays/tensors; bool
        Returns:
            dict: dictionary containing the metrics; dict
        """
        # copy input arrays/tensors
        if copy:
            data = self.__copyData(data=data)

        # check that all required data is provided
        self.__checkData(data=data, eval_metrics=eval_metrics)

        # convert data to right format and coordinate system
        if 'depth' in data: # TODO: change condition and function
            data = self.convertData(data=data, eval_metrics=eval_metrics, convert_to_world_coords=convert_to_world_coords)

        dict = {}
        for metric in eval_metrics:
            if metric == 'rmse':
                dict['rmse'] = self.__rmse(depth=data['depth'], depth_gt=data['depth_gt'])
            elif metric == 'mae':
                dict['mae'] = self.__mae(depth=data['depth'], depth_gt=data['depth_gt'])
            elif metric == 'mare':
                dict['mare'] = self.__mare(depth=data['depth'], depth_gt=data['depth_gt'])
            elif metric == 'nn':
                nn_dists, mnn = self.__nn(pos=data['pos'], pos_gt=data['pos_gt'])
                dict['nn_dists'] = nn_dists
                dict['mnn'] = mnn
            elif metric == 'psnr':
                dict['psnr'] = self.__psnr(rgb=data['rgb'], rgb_gt=data['rgb_gt'])
            elif metric == 'ssim':
                dict['ssim'] = self.__ssim(rgb=data['rgb'], rgb_gt=data['rgb_gt'])
            else:
                print(f"WARNING: metric {metric} not implemented")

        return dict
    
    def nnNumpy(
            self, 
            array1:np.array, 
            array2:np.array
        ):
        """
        Find the closest points in array2 for each point in array1
        and return the indices of array2 for each point in array1.
        Args:
            array1: array of float (N, 2/3)
            array2: array of float (M, 2/3)
        Returns:
            nn_idxs: indices of nearest neighbours from array2 with respect to array1; array of int (N,)
            nn_dists: distances of nearest neighbours from array2 with respect to array1; array of float (N,)
        """
        # downsample arrays
        array1 = np.copy(array1.astype(np.float32))
        array2 = np.copy(array2.astype(np.float32))

        # determine nearest neighbour indices and distances
        dists = np.linalg.norm(array2[:, np.newaxis] - array1, axis=2) # (M, N)
        nn_idxs = np.argmin(dists, axis=0) # (N,)
        nn_dists = np.min(dists, axis=0) # (N,)
        
        return nn_idxs, nn_dists
    
    def nnTorch(
            self, 
            tensor1:torch.tensor, 
            tensor2:torch.tensor,
        ):
        """
        Find the closest points in array2 for each point in array1
        and return the indices of array2 for each point in array1.
        Args:
            array1: tensor of float (N, 2/3)
            array2: tensor of float (M, 2/3)
        Returns:
            nn_idxs: indices of nearest neighbours from tensor2 with respect to tensor1; array of int (N,)
            nn_dists: distances of nearest neighbours from tensor2 with respect to tensor1; array of float (N,)
        """
        # downsample arrays
        tensor1 = np.copy(tensor1.astype(np.float32))
        tensor2 = np.copy(tensor2.astype(np.float32))

        # determine nearest neighbour indices and distances
        dists = torch.linalg.norm(tensor2[:, np.newaxis] - tensor1, dim=2) # (M, N)
        nn_idxs = torch.argmin(dists, dim=0) # (N,)
        nn_dists = torch.min(dists, dim=0) # (N,)
        
        return nn_idxs, nn_dists
    
    def __rmse(
            self,
            depth, 
            depth_gt
        ):
        """
        Calculate Root Mean Squared Error (RMSE) between depth and depth_gt
        Args:
            depth: predicted depth; either numpy array or torch tensor (N*M,)
            depth_gt: ground truth depth; either numpy array or torch tensor (N*M,)
        Returns:
            rmse: root mean squared error; float
        """
        if torch.is_tensor(depth):
            return torch.sqrt(torch.nanmean((depth - depth_gt)**2)).item()
        return np.sqrt(np.nanmean((depth - depth_gt)**2))

    def __mae(
            self, 
            depth, 
            depth_gt
        ):
        """
        Calculate Mean Absolute Error (MAE) between depth and depth_gt
        Args:
            depth: predicted depth; either numpy array or torch tensor (N*M,)
            depth_gt: ground truth depth; either numpy array or torch tensor (N*M,)
        Returns:
            mae: mean absolute error; float
        """
        if torch.is_tensor(depth):
            return torch.nanmean(torch.abs(depth - depth_gt)).item()
        return np.nanmean(np.abs(depth - depth_gt))
    
    def __mare(
            self, 
            depth, 
            depth_gt
        ):
        """
        Calculate Mean Absolute Relative Error (MARE) between depth and depth_gt
        Args:
            depth: predicted depth; either numpy array or torch tensor (N*M,)
            depth_gt: ground truth depth; either numpy array or torch tensor (N*M,)
        Returns:
            mare: mean absolute relative error; float
        """
        if torch.is_tensor(depth):
            return torch.nanmean(torch.abs((depth - depth_gt)/ depth_gt)).item()
        return np.nanmean(np.abs((depth - depth_gt)/ depth_gt))
    
    def __nn(
            self, 
            pos, 
            pos_gt
        ):
        """
        Calculate nearest neighbour distance between pos_w and pos_w_gt
        Args:
            pos: predicted position in world coordinate system; either numpy array or torch tensor (N, M, 2)
            pos_gt: ground truth position in world coordinate system; either numpy array or torch tensor (N, M, 2)
        Returns:
            nn_dists: nearest neighbour distances; either numpy array or torch tensor (N,)
            mnn: mean of nearest neighbour distances; float
        """
        if torch.is_tensor(pos):
            nn_dists = torch.zeros_like(pos)
        else:
            nn_dists = np.zeros_like(pos)
        
        for i in range(pos.shape[0]): # TODO: pos should have shape (N, M, 2)
            if torch.is_tensor(pos):
                _, dists = self.nnTorch(tensor1=pos_gt[i], tensor2=pos[i])
            else:
                _, dists = self.nnNumpy(array1=pos_gt[i], array2=pos[i])
            nn_dists[i] = dists

        if torch.is_tensor(pos):
            mnn = torch.mean(nn_dists).item()
        else:
            mnn = np.mean(nn_dists)

        return nn_dists, mnn
    

    def __psnr(
            self,
            rgb:torch.tensor,
            rgb_gt:torch.tensor,
    ):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between rgb and rgb_gt
        Args:
            rgb: predicted rgb; either numpy array or torch tensor (N*H*W, 3)
            rgb_gt: ground truth rgb; either numpy array or torch tensor (N*H*W, 3)
        Returns:
            psnr: average peak signal-to-noise ratio; float
        """
        W, H = self.img_wh
        if rgb.shape[0] % (W*H) != 0:
            print(f"ERROR: metric:_psnr: rgb.shape[0] = {rgb.shape[0]} must be divisible by W*H = {W*H}")
        num_imgs = rgb.shape[0] // (W*H)

        test_psnrs = []
        for i in range(num_imgs):
            rgb_img = rearrange(rgb[i*H*W:(i+1)*W*H], '(h w) c -> 1 c h w', h=H) # TODO: optimize
            rgb_img_gt = rearrange(rgb_gt[i*H*W:(i+1)*W*H], '(h w) c -> 1 c h w', h=H)

            self.val_psnr(rgb_img, rgb_img_gt)
            test_psnrs.append(self.val_psnr.compute())
            self.val_psnr.reset()

        return np.sum(test_psnrs) / len(test_psnrs)

    def __ssim(
            self,
            rgb:torch.tensor,
            rgb_gt:torch.tensor,
    ):
        """
        Calculate Structural Similarity Index Measure (SSIM) between rgb and rgb_gt.
        Args:
            rgb: predicted rgb; either numpy array or torch tensor (N*H*W, 3)
            rgb_gt: ground truth rgb; either numpy array or torch tensor (N*H*W, 3)
        Returns:
            ssim: average structural similarity index measure; float
        """
        W, H = self.img_wh
        if rgb.shape[0] % (W*H) != 0:
            print(f"ERROR: metric:_psnr: rgb.shape[0] = {rgb.shape[0]} must be divisible by W*H = {W*H}")
        num_imgs = rgb.shape[0] // (W*H)

        test_ssims = []
        for i in range(num_imgs):
            rgb_img = rearrange(rgb[i*H*W:(i+1)*W*H], '(h w) c -> 1 c h w', h=H) # TODO: optimize
            rgb_img_gt = rearrange(rgb_gt[i*H*W:(i+1)*W*H], '(h w) c -> 1 c h w', h=H)

            self.val_ssim(rgb_img, rgb_img_gt)
            test_ssims.append(self.val_ssim.compute())
            self.val_ssim.reset()

        return np.sum(test_ssims) / len(test_ssims)
    
    def __copyData(
            self,
            data:dict,
    ):
        """
        Copy data dictionary.
        Args:
            data: data dictionary
        Returns:
            data_copy: data dictionary
        """
        data_copy = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                data_copy[key] = value.clone().detach()
            else:
                data_copy[key] = np.copy(value)
        return data_copy
    
    def __checkData(
            self,
            data:dict,
            eval_metrics:list,
    ):
        """
        Check if data dictionary contains all required keys.
        Args:
            data: data dictionary
            eval_metrics: list of metrics to evaluate; list of str
        """
        if 'nn' in eval_metrics:
            if (not 'rays_o' in data) or (not 'scan_angles' in data):
                print("WARNING: rays_o and scan_angles must be provided for metric 'nn'")
                eval_metrics.remove('nn')

        if ('nn' in eval_metrics) or ('mse' in eval_metrics) or ('mae' in eval_metrics) or ('mare' in eval_metrics):
            if (not 'depth' in data) or (not 'depth_gt' in data):
                print("WARNING: pos must be provided for metrics 'nn', 'mse', 'mae', 'mare'")
                eval_metrics.remove('nn')
                eval_metrics.remove('mse')
                eval_metrics.remove('mae')
                eval_metrics.remove('mare')

        if ('psnr' in eval_metrics) or ('ssim' in eval_metrics):
            if (not 'rgb' in data) or (not 'rgb_gt' in data):
                print("WARNING: rgb and rgb_gt must be provided for metrics 'psnr', 'ssim'")
                eval_metrics.remove('psnr')
                eval_metrics.remove('ssim')

    