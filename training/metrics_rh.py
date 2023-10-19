import numpy as np
import torch

from datasets.robot_at_home_scene import RobotAtHomeScene
from training.metrics import Metrics
from args.args import Args

class MetricsRH(Metrics):
    def __init__(
            self,
            args:Args,
            rh_scene:RobotAtHomeScene,
            img_wh:tuple,
        ) -> None:
        """
        Metrics base class.
        Args:
            args: Args object
            rh_scene: RobotAtHomeScene object
            img_wh: image width and height; tuple
        """
        Metrics.__init__(self, args=args, img_wh=img_wh)
        self.rh_scene = rh_scene

    def convertData(
            self, 
            data:dict,
            num_test_pts:int,
            eval_metrics:list,
            convert_to_world_coords:bool,
        ):
        """
        Convert data to world coordinates and get position of depths in world coordinates.
        Args:
            data: data dictionary
            num_test_pts: number of test points; int
            eval_metrics: evaluation metrics; list of str
            convert_to_world_coords: convert depth to world coordinates (meters); bool
        Returns:
            data: data dictionary
        """
        depth = data['depth']
        depth_gt = data['depth_gt']
        rays_o = data['rays_o']
        scan_angles = data['scan_angles']

        # convert depth to world coordinates (meters)
        if convert_to_world_coords:
            depth = self.rh_scene.c2wTransformation(pos=depth, only_scale=True, copy=False)
            depth_gt = self.rh_scene.c2wTransformation(pos=depth_gt, only_scale=True, copy=False)
            if rays_o is not None:
                rays_o = self.rh_scene.c2wTransformation(pos=rays_o, copy=False)

        # convert depth to position in world coordinate system
        if 'nn' in eval_metrics:
            if torch.is_tensor(depth):
                depth = depth.clone().detach().numpy()
                depth_gt = depth_gt.clone().detach().numpy()
            pos = self.rh_scene.convertDepth2Pos(rays_o, depth, scan_angles)
            pos_gt = self.rh_scene.convertDepth2Pos(rays_o, depth_gt, scan_angles)
            pos = pos.reshape(num_test_pts, -1, 2)
            pos_gt = pos_gt.reshape(num_test_pts, -1, 2)

        data['depth'] = depth
        data['depth_gt'] = depth_gt
        if 'nn' in eval_metrics:
            data['pos'] = pos
            data['pos_gt'] = pos_gt
        return data