import torch
import os

class Args():
    def __init__(self, hparams):
        # general
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 23

        # saving and loading
        self.val_dir = 'results/'
        # check if val_dir exists, otherwise create it
        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)

        # rendering configuration
        self.exp_step_factor = 1 / 256 if hparams.scale > 0.5 else 0. 