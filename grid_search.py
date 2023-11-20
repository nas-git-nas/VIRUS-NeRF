import numpy as np

from args.args import Args
from training.trainer import Trainer
from modules.occupancy_grid import OccupancyGrid
from datasets.dataset_rh import DatasetRH


def loopOccgrid(
    params:dict,
):
    """
    Loop through all possible combinations of parameters
    Args:
        params: dictionary of parameters to loop through; dict
    Yields:
        param: dictionary of parameters for each iteration; dict
    """
    for update_interval in params["update_interval"]:
        for decay_warmup_steps in params["decay_warmup_steps"]:
             for batch_ratio_ray_update in params["batch_ratio_ray_update"]:
                  for false_detection_prob_every_m in params["false_detection_prob_every_m"]:
                       for std_every_m in params["std_every_m"]:
                            for nerf_threshold_max in params["nerf_threshold_max"]:
                                 for nerf_threshold_slope in params["nerf_threshold_slope"]:
                                      yield {
                                            "update_interval": update_interval,
                                            "decay_warmup_steps": decay_warmup_steps,
                                            "batch_ratio_ray_update": batch_ratio_ray_update,
                                            "false_detection_prob_every_m": false_detection_prob_every_m,
                                            "std_every_m": std_every_m,
                                            "nerf_threshold_max": nerf_threshold_max,
                                            "nerf_threshold_slope": nerf_threshold_slope,
                                        }


def searchOccgrid():
    """
    Occupancy grid grid search
    """
    # define parameters
    params = {
        "update_interval": [8],
        "decay_warmup_steps": [10],
        "batch_ratio_ray_update": [0.5],
        "false_detection_prob_every_m": [0.3],
        "std_every_m": [0.1, 0.2, 0.3],
        "nerf_threshold_max": [1, 5.91, 10],
        "nerf_threshold_slope": [0.1, 0.01, 0.001],
    }
    seeds = [23, 42, 69]
    hparams_file = "rh_windows.json"

    # get hyper-parameters and other variables
    args = Args(
        file_name=hparams_file
    )

    # datasets   
    if args.dataset.name == 'robot_at_home':
        dataset = DatasetRH    
    train_dataset = dataset(
        args = args,
        split="train",
    ).to(args.device)
    test_dataset = dataset(
        args = args,
        split='test',
        scene=train_dataset.scene,
    ).to(args.device)
    
    # evaluate all parameter combinations
    for i, param in enumerate(loopOccgrid(params)):
        print("\n\n----- NEW PARAMETERS -----")
        print(f"iteration: {i}, param: {param}")

        for seed in seeds:

            # set param
            args.setRandomSeed(
                seed=seed,
            )
            for key, value in param.items():
                setattr(args.occ_grid, key, value)
            if i != 0:
                trainer.args.createSaveDir()

            # load trainer
            trainer = Trainer(
                args=args,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )

            # train and evaluate
            trainer.train()
            trainer.evaluate()

def evaluateSearch():
    """
    Evaluate grid search
    """
    grid_search_dir = "results/robot_at_home/20210602_183722"

    

if __name__ == "__main__":
    searchOccgrid()
