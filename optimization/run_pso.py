import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import time
import torch
import taichi as ti

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
    hparams_file = "ethz_usstof_win.json" 
    hparams_lims_file = "optimization/hparams_lims.json"
    save_dir = "results/pso/opt3"

    # get hyper-parameters and other variables
    args = Args(
        file_name=hparams_file
    )
    args.eval.eval_every_n_steps = args.training.max_steps + 1
    args.eval.plot_results = False
    args.model.save = False
    args.eval.sensors = ["GT", "NeRF"]

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
        hparams_dict = pso.nextHparams(
            group_dict_layout=True,
            name_dict_layout=False,
        ) # np.array (M,)

        print("\n\n----- NEW PARAMETERS -----")
        print(f"Time: {time.time()-pso.start_time:.1f}/{T}, param: {hparams_dict}")
        print(f"Current best mnn: {np.min(pso.best_score):.3f}")

        # set hparams
        args.setRandomSeed(
            seed=args.seed+iter,
        )
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

        # save state
        pso.saveState(
            score=metrics_dict['NeRF']["nn_mean"],
        )

        # update particle swarm
        terminate = pso.update(
            score=metrics_dict['NeRF']["nn_mean"],
        ) # bool

        # watch memory usage
        if args.device == "cuda":
            nvidia_smi.nvmlInit()

            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) # gpu id 0
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            print(f"Free memory: {info.free}/{info.total}")
            if info.used > 11e9:
                print("Run PSO: Used memory is too high. Exiting...")
                terminate = True

            nvidia_smi.nvmlShutdown()


if __name__ == "__main__":
    main()