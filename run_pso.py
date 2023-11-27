import numpy as np
import time
import sys

from optimization.particle_swarm_optimization import ParticleSwarmOptimization
from args.args import Args
from datasets.dataset_rh import DatasetRH
from training.trainer import Trainer
# from helpers.system_fcts import get_size

def get_size(
    obj, 
    seen=None
):
    """
    Recursively finds size of objects.
    Source: https://goshippo.com/blog/measure-real-size-any-python-object/
    Args:
        obj: object to find size of
        seen: helper object to keep track of seen objects
    Returns:
        size of object in bytes
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def main():
    # define paraeters
    T_time = 36000 # seconds
    hparams_file = "rh_gpu.json" # "rh_gpu.json"
    hparams_lims_file = "optimization/hparams_lims.json"
    save_dir = "results/pso/opt3"

    # get hyper-parameters and other variables
    args = Args(
        file_name=hparams_file
    )
    args.eval.eval_every_n_steps = args.training.max_steps + 1
    args.eval.plot_results = False
    args.model.save = False

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

    # pso
    pso = ParticleSwarmOptimization(
        hparams_lims_file=hparams_lims_file,
        save_dir=save_dir,
        T_iter=None,
        T_time=T_time,
        rng=np.random.default_rng(args.seed),
    )

    # run optimization
    terminate = False
    while not terminate:
        # get hparams to evaluate
        hparams_dict = pso.getHparams(
            group_dict_layout=True,
        ) # np.array (M,)

        print("\n\n----- NEW PARAMETERS -----")
        print(f"Time: {time.time()-pso.start_time:.1f}/{T_time}, param: {hparams_dict}")
        print(f"Current best mnn: {np.min(pso.best_score):.3f}")

        # set hparams
        args.setRandomSeed(
            seed=args.seed+1,
        )
        # for key, value in hparams_dict["occ_grid"].items():
        #     setattr(args.occ_grid, key, value)

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
            score=metrics_dict["mnn"],
        )

        # update particle swarm
        terminate = pso.update(
            score=metrics_dict["mnn"],
        ) # bool

        print(f"References to trainer: {get_size(trainer)}")
        print(f"Size of trainer: {get_size(trainer)}")
        print(f"Size of args: {get_size(args)}")
        print(f"Size of train_dataset: {get_size(train_dataset)}")
        print(f"Size of test_dataset: {get_size(test_dataset)}")
        print(f"Size of PSO: {get_size(pso)}")
        del trainer


if __name__ == "__main__":
    main()