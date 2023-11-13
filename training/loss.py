import torch
import torch.nn.functional as F

from args.args import Args
from datasets.scene_base import SceneBase

class Loss():
    def __init__(
            self,
            args:Args,
            scene:SceneBase=None,
            sensors_dict:dict=None,
    ) -> None:
        """
        Loss class
        Args:
            args: arguments from command line
            rh_scene: RobotAtHomeScene object
            sensors_dict: dict containint SensorModel objects, dict
        """
        self.args = args

        # log sub-losses in dictionary
        self.loss_dict = {}
        self.log_loss = False
        self.step = 0

        if 'USS' in self.args.training.sensors:
            uss_depth_tol = scene.w2c(pos=0.03, only_scale=True, copy=True)
            self.uss_depth_tol = torch.tensor(uss_depth_tol, device=self.args.device, dtype=torch.float32)

            self.sensors_dict = sensors_dict

    def __call__(
            self,
            results:dict,
            data:dict,
            return_loss_dict:bool=False,
    ):
        """
        Loss function used during training
        Args:
            results: dict of rendered images
                'opacity': sum(transmittance*alpha); array of shape: (N,)
                'depth': sum(transmittance*alpha*t_i); array of shape: (N,)
                'rgb': sum(transmittance*alpha*rgb_i); array of shape: (N, 3)
                'total_samples': total samples for all rays; int
                where   transmittance = exp( -sum(sigma_i * delta_i) )
                        alpha = 1 - exp(-sigma_i * delta_i)
                        delta_i = t_i+1 - t_i
            data: dict of ground truth images
                'img_idxs': image indices; array of shape (N,) or (1,) if same image
                'pix_idxs': pixel indices; array of shape (N,)
                'pose': poses; array of shape (N, 3, 4)
                'direction': directions; array of shape (N, 3)
                'rgb': pixel colours; array of shape (N, 3)
                'depth': dict containing pixel depths of different sensors; {str: tensor of shape (N,)}
            return_loss_dict: whether to return loss dictionary; bool
        Returns:
            total_loss: loss value; tensor of float (1,)
            loss_dict: dict of loss values; dict of float
        """
        self.step += 1
        self.loss_dict = {}
        if return_loss_dict:
            self.log_loss = True
        else:
            self.log_loss = False

        # calculate losses
        color_loss = self._colorLoss(results=results, data=data)
        depth_loss = self._depthLoss(results=results, data=data)
        total_loss = color_loss + depth_loss * self.args.training.depth_loss_w

        if self.log_loss:
            self.loss_dict['total'] = total_loss.item()
        return total_loss, self.loss_dict

    def _colorLoss(
            self, 
            results:dict, 
            data:dict,
    ):
        """
        Loss function for training
        Args:
            results: dict of rendered images
            data: dict of ground truth images
        Returns:
            color_loss: colour loss value; tensor of float (1,)
        """
        color_loss = F.mse_loss(results['rgb'], data['rgb'])

        if self.log_loss:
            self.loss_dict['color'] = color_loss.item()
        return color_loss

    def _depthLoss(
            self,
            results:dict,
            data:dict,
    ):
        """
        Depth loss depending on sensor model.
        Args:
            results: dict of rendered images
            data: dict of ground truth images
        Returns:
            depth_loss: depth loss value; tensor of float (1,)
        """
        depth_loss = torch.tensor(0.0, device=self.args.device, dtype=torch.float32)
        for sensor_name in self.args.training.sensors:
            if sensor_name == 'RGBD':
                depth_loss += self._depthLossRGBD(results=results, data=data)
            elif sensor_name == 'ToF':
                depth_loss += self._depthLossToF(results=results, data=data)
            elif sensor_name == 'USS':
                depth_loss += self._depthLossUSS(results=results, data=data)
            else:
                self.args.logger.error(f"sensor name '{sensor_name}' is invalid")
        depth_loss /= len(self.args.training.sensors)
        
        if self.log_loss:
            self.loss_dict['depth'] = depth_loss.item() * self.args.training.depth_loss_w
        return depth_loss

    def _depthLossRGBD(
            self, 
            results:dict, 
            data:dict,
    ):
        """
        Loss function for RGBD sensor model
        Args:
            results: dict of rendered images
            data: dict of ground truth images
        Returns:
            depth_loss: depth loss value; tensor of float (1,)
        """
        val_idxs = ~torch.isnan(data['depth']['RGBD'])
        rgbd_loss = F.mse_loss(results['depth'][val_idxs], data['depth']['RGBD'][val_idxs])

        if self.log_loss:
            self.loss_dict['rgbd'] = rgbd_loss.item() * self.args.training.depth_loss_w
        return rgbd_loss
        
    def _depthLossToF(
            self, 
            results:dict, 
            data:dict,
    ):
        """
        Loss function for ToF sensor model
        Args:
            results: dict of rendered images
            data: dict of ground truth images
        Returns:
            depth_loss: depth loss value; tensor of float (1,)
        """
        val_idxs = ~torch.isnan(data['depth']['ToF'])
        tof_loss = F.mse_loss(results['depth'][val_idxs], data['depth']['ToF'][val_idxs])

        if self.log_loss:
            self.loss_dict['ToF'] = tof_loss.item() * self.args.training.depth_loss_w
        return tof_loss

    def _depthLossUSS(
            self, 
            results:dict, 
            data:dict,
    ):
        """
        Loss function for USS sensor model
        Args:
            results: dict of rendered images
            data: dict of ground truth images
        Returns:
            depth_loss: depth loss value; tensor of float (1,)
        """ 
        # get minimum depth per image for batch 
        imgs_depth_min, weights = self.sensors_dict['USS'].updateDepthMin(
            results=results,
            data=data,
        ) # (num_test_imgs,), (num_test_imgs,)
        depths_min = imgs_depth_min[data['img_idxs']] # (N,)
        weights = weights[data['img_idxs']] # (N,)

        # mask data
        uss_mask = ~torch.isnan(data['depth']['USS'])
        depth_mask = results['depth'] < depths_min + self.uss_depth_tol  
        close_mask = results['depth'] < data['depth']['USS'] - self.uss_depth_tol  

        # calculate close loss: pixels that are closer than the USS measurement
        uss_loss_close = torch.tensor(0.0, device=self.args.device, dtype=torch.float32)
        if torch.any(uss_mask & close_mask):
            uss_loss_close = torch.mean(
                (results['depth'][uss_mask & close_mask] - data['depth']['USS'][uss_mask & close_mask])**2
            )

        # calculate min loss: error of minimal depth wrt. USS measurement
        uss_loss_min = torch.tensor(0.0, device=self.args.device, dtype=torch.float32)
        if torch.any(uss_mask & depth_mask):
            uss_loss_min = torch.mean(
                weights[uss_mask & depth_mask] 
                * (results['depth'][uss_mask & depth_mask] - data['depth']['USS'][uss_mask & depth_mask])**2
            )  

        # if self.step%25 == 0:
        #     print(f"depth mask sum: {torch.sum(uss_mask & depth_mask)}, close mask sum: {torch.sum(uss_mask & close_mask)}, weights mean: {torch.mean(weights):.3f}")
        #     print(f"min_loss: {uss_loss_min:.5f} | close_loss: {uss_loss_close:.5f}")

        uss_loss = uss_loss_close + uss_loss_min
        if self.log_loss:
            self.loss_dict['USS'] = uss_loss.item() * self.args.training.depth_loss_w
            self.loss_dict['USS_close'] = uss_loss_close.item() * self.args.training.depth_loss_w
            self.loss_dict['USS_min'] = uss_loss_min.item() * self.args.training.depth_loss_w
        return uss_loss
    
    
