import torch
import os
import json
from datetime import datetime
import shutil

from h_params import HParamsDataset, HParamsModel, HParamsTraining, HParamsOccGrid


class Args():
    def __init__(self, file_name) -> None:
        # hyper parameters
        self.dataset = HParamsDataset()
        self.model = HParamsModel()
        self.training = HParamsTraining()
        self.occ_grid = HParamsOccGrid()

        # set hyper parameters
        hparams = self.readJson(file_name)
        self.dataset.setHParams(hparams)
        self.model.setHParams(hparams)
        self.training.setHParams(hparams)
        self.occ_grid.setHParams(hparams)

        # general
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 23

        # create saving directory
        t = datetime.now()
        time_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M")
        self.save_dir = os.path.join('results/', self.dataset.name, time_name)
        if not os.path.exists('results/'):
            os.mkdir('results/')
        if not os.path.exists(os.path.join('results/', self.dataset.name)):
            os.mkdir(os.path.join('results/', self.dataset.name))
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.mkdir(self.save_dir)

        # rendering configuration
        self.exp_step_factor = 1 / 256 if self.model.scale > 0.5 else 0. 


    def readJson(self, file_name):
        """
        Read hyper parameters from json file
        Args:
            file_name: name of json file; str
        Returns:
            hparams: hyper parameters; dict
        """
        file_path = os.path.join("args", file_name)
        with open(file_path) as f:
            hparams = json.load(f)

        return hparams
    
    def saveJson(self):
        """
        Save arguments in json file
        """
        hparams = {}
        hparams["dataset"] = self.dataset.getHParams()
        hparams["model"] = self.model.getHParams()
        hparams["training"] = self.training.getHParams()
        hparams["occ_grid"] = self.occ_grid.getHParams()
        
        # Serializing json
        json_object = json.dumps(hparams, indent=4)
        
        # Writing to sample.json
        with open(os.path.join("results", "hparams.json"), "w") as outfile:
            outfile.write(json_object)


def test_args():
    args = Args("hparams.json")
    args.saveJson()

if __name__ == "__main__":
    test_args()
