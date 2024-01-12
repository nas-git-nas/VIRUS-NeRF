import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import time
import torch
import taichi as ti
from icecream import ic

from optimization.particle_swarm_optimization_wrapper import ParticleSwarmOptimizationWrapper
from args.args import Args
from datasets.dataset_rh import DatasetRH
from datasets.dataset_ethz import DatasetETHZ
from training.trainer import Trainer
from helpers.system_fcts import get_size, moveToRecursively

if torch.cuda.is_available():
    import nvidia_smi

def main():
    # define paraeters
    T = 36000 # if termination_by_time: T is time in seconds, else T is number of iterations
    termination_by_time = True # whether to terminate by time or iterations
    hparams_file = "ethz_usstof_gpu.json" 
    hparams_lims_file = "optimization/hparams_lims.json"
    save_dir = "results/pso/opt16"

    # get hyper-parameters and other variables
    args = Args(
        file_name=hparams_file
    )
    args.eval.eval_every_n_steps = args.training.max_steps + 1
    args.eval.plot_results = False
    args.model.save = False
    args.eval.sensors = ["GT", "NeRF"]
    args.eval.num_color_pts = 0
    args.eval.batch_size = 8192
    args.training.batch_size = 4096

    # datasets   
    if args.dataset.name == 'RH2':
        dataset = DatasetRH
    elif args.dataset.name == 'ETHZ':
        dataset = DatasetETHZ
    else:
        args.logger.error("Invalid dataset name.")    
    train_dataset = dataset(
        args = args,
        split="train",
    ).to(args.device)
    test_dataset = dataset(
        args = args,
        split='test',
        scene=train_dataset.scene,
    ).to(args.device)

    # pso
    pso = ParticleSwarmOptimizationWrapper(
        hparams_lims_file=hparams_lims_file,
        save_dir=save_dir,
        T=T,
        termination_by_time=termination_by_time,
        rng=np.random.default_rng(args.seed),
    )

    # run optimization
    terminate = False
    iter = 0
    while not terminate:
        iter += 1

        # get hparams to evaluate
        hparams_dict = pso.getNextHparams(
            group_dict_layout=True,
            name_dict_layout=False,
        ) # np.array (M,)

        print("\n\n----- NEW PARAMETERS -----")
        print(f"Time: {time.time()-pso.time_start:.1f}/{T}, particle: {pso.n}")
        ic(hparams_dict)
        print(f"Current best mnn: {np.min(pso.best_score):.3f}, best particle: {np.argmin(pso.best_score)}")

        # set hparams
        args.setRandomSeed(
            seed=args.seed+iter,
        )

        sampling_strategy = {
            "imgs": "all", 
            "pixs": {
                "closest": hparams_dict["training"]["pixs_closest"],
                "valid_uss": hparams_dict["training"]["pixs_valid_uss"],
                "valid_tof": 1 - hparams_dict["training"]["pixs_valid_uss"] \
                               - hparams_dict["training"]["pixs_closest"],
            },
        }
        setattr(args.training, "sampling_strategy", sampling_strategy)

        for key, value in hparams_dict["training"].items():
            if "pixs" in key:
                continue        
            setattr(args.training, key, value)

        for key, value in hparams_dict["occ_grid"].items():
            setattr(args.occ_grid, key, value)

        # load trainer
        trainer = Trainer(
            args=args,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

        # train and evaluate model
        trainer.train()
        metrics_dict = trainer.evaluate()

        # get score
        score = metrics_dict['NeRF']["nn_mean"]['zone3']
        if score == np.nan:
            score = np.inf

        # update particle swarm
        terminate = pso.update(
            score=score,
        ) # bool

        # save state
        pso.saveState(
            score=score,
        )

        del trainer
        # watch memory usage
        if torch.cuda.is_available():
            nvidia_smi.nvmlInit()

            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) # gpu id 0
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            print(f"Run PSO: Free memory: {(info.free/1e6):.2f}Mb / {(info.total/1e6):.2f}Mb = {(info.free/info.total):.3f}%")
            if info.free < 2e9:
                print("Run PSO: Used memory is too high. Exiting...")
                terminate = True

            nvidia_smi.nvmlShutdown()


if __name__ == "__main__":
    main()