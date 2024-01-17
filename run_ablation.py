
import os
import numpy as np
import sys

from training.trainer import Trainer
from args.args import Args
from datasets.dataset_ethz import DatasetETHZ
from helpers.system_fcts import checkGPUMemory

def main():
    hparams_file = "ethz_usstof_opt_gpu.json"
    num_trainings = 5
    base_dir = "results/ETHZ/ablation/best_particle"

    # create base dir and count seeds already trained
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    else:
        print("Base dir already exists.")
        sys.exit()
    num_seeds_already_trained = len(os.listdir(base_dir))

    # args
    args = Args(
        file_name=hparams_file
    )

    # datasets      
    train_dataset = DatasetETHZ(
        args = args,
        split="train",
    ).to(args.device)
    test_dataset = DatasetETHZ(
        args = args,
        split='test',
        scene=train_dataset.scene,
    ).to(args.device)

    for i in range(num_seeds_already_trained, num_trainings):
        # set random seed and directory for saving
        args.setRandomSeed(
            seed=args.seed+i,
        )
        args.save_dir = os.path.join(base_dir, f"seed_{args.seed}")
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        else:
            print(f"Seed {args.seed} already trained.")
            sys.exit()

        # create trainer
        trainer = Trainer(
            args=args,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
        trainer.train()
        trainer.evaluate()

        # check if GPU memory is full
        if checkGPUMemory():
            break

if __name__ == "__main__":
    main()


