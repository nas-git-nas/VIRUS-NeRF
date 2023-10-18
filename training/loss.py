import torch
import torch.nn.functional as F

from args.args import Args

class Loss():
    def __init__(
            self,
            args:Args,
    ) -> None:
        """
        Loss class
        Args:
            args: arguments from command line
        """
        self.args = args

        # log sub-losses in dictionary
        self.loss_dict = {}
        self.log_loss = False

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
                'depth': pixel depths; array of shape (N,)
            return_loss_dict: whether to return loss dictionary; bool
        Returns:
            total_loss: loss value; tensor of float (1,)
            loss_dict: dict of loss values; dict of float
        """
        if return_loss_dict:
            self.log_loss = True
            self.loss_dict = {}
        else:
            self.log_loss = False

        color_loss = self._colorLoss(results=results, data=data)
        depth_loss = self._depthLoss(results=results, data=data)
        
        depth_loss = depth_loss * self.args.training.depth_loss_w
        total_loss = color_loss + depth_loss
        return total_loss, color_loss, depth_loss

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
            self.loss_dict['colour_loss'] = color_loss.item()

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
        if self.args.rh.sensor_model == 'RGBD':
            return self._depthLossRGBD(results=results, data=data)
        if self.args.rh.sensor_model == 'ToF':
            return self._depthLossToF(results=results, data=data)
        if self.args.rh.sensor_model == 'USS':
            return self._depthLossUSS(results=results, data=data)
        
        return torch.tensor(0.0, device=self.args.device, dtype=torch.float32)

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
        val_idxs = ~torch.isnan(data['depth'])
        return F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])
        
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

        if self.args.rh.sensor_model == 'RGBD' or self.args.rh.sensor_model == 'ToF':
            val_idxs = ~torch.isnan(data['depth'])
            return F.mse_loss(results['depth'][val_idxs], data['depth'][val_idxs])

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
        imgs_depth_min, weights = self.train_dataset.sensor_model.updateDepthMin(
            results=results,
            data=data,
            step=step,
        ) # (num_test_imgs,), (num_test_imgs,)
        depths_min = imgs_depth_min[data['img_idxs']] # (N,)
        weights = weights[data['img_idxs']] # (N,)

        # mask data
        depth_tolerance = self.train_dataset.scene.w2cTransformation(pos=0.03, only_scale=True, copy=True)
        depth_tolerance = torch.tensor(depth_tolerance, device=self.args.device, dtype=torch.float32)
        uss_mask = ~torch.isnan(data['depth'])
        depth_mask = results['depth'] < depths_min + depth_tolerance  
        close_mask = results['depth'] < data['depth'] - depth_tolerance  

        # calculate loss
        depth_loss = torch.tensor(0.0, device=self.args.device, dtype=torch.float32)

        depth_data = data['depth'][uss_mask & depth_mask]
        w = weights[uss_mask & depth_mask]
        depth_results = results['depth'][uss_mask & depth_mask]
        if torch.any(uss_mask & depth_mask):
            min_loss = torch.mean(w * (depth_results-depth_data)**2)
            depth_loss += min_loss
            if step%25 == 0:
                print(f"min_loss: {min_loss}")

        depth_data = data['depth'][uss_mask & close_mask]
        depth_results = results['depth'][uss_mask & close_mask]
        if torch.any(uss_mask & close_mask):
            close_loss = torch.mean((depth_results-depth_data)**2)
            depth_loss += close_loss
            if step%25 == 0:
                print(f"close_loss: {close_loss}")

        if step%25 == 0:
            print(f"depth mask sum: {torch.sum(uss_mask & depth_mask)}, close mask sum: {torch.sum(uss_mask & close_mask)}, weights mean: {torch.mean(weights)}")
        
        return depth_loss
    
    
