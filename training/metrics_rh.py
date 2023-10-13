import numpy as np
import torch

from datasets.robot_at_home_scene import RobotAtHomeScene

class Metrics():
    def __init__(
            self, 
            rh_scene:RobotAtHomeScene,
            eval_metrics:list=['rmse', 'mae', 'mare'],
        ) -> None:
        """
        Metrics base class.
        Args:
            rh_scene: RobotAtHomeScene object
            eval_metrics: list of metrics to evaluate; list of str
        """
        self.rh_scene = rh_scene
        self._eval_metrics = np.copy(eval_metrics)