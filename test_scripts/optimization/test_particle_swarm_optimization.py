import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from abc import ABC, abstractmethod
 
sys.path.insert(0, os.getcwd())
from optimization.particle_swarm_optimization_wrapper import ParticleSwarmOptimizationWrapper
from optimization.metric import Metric
from optimization.plotter import Plotter


def optimize(
    pso:ParticleSwarmOptimizationWrapper,
    metric:Metric,
):
    """
    Run optimization.
        N: number of particles; int
        M: number of hparams; int
        T: number of iterations; int
    Args:
        pso: particle swarm optimization; ParticleSwarmOptimizationWrapper
        metric: metric to optimize; Metric
    """
    terminate = False
    while not terminate:
        # get hparams to evaluate
        X = pso.nextHparams(
            group_dict_layout=False,
            name_dict_layout=False,
        ) # np.array (M,)
        print(f"Iteration: t: {pso.t}, param: {X}")

        # evaluate metric
        score = metric(
            X=X,
        ) # float

        # save state
        pso.saveState(
            score=score,
        )

        # update particle swarm
        terminate = pso.update(
            score=score,
        ) # bool

def test_pso():
    # define optimization algorithm
    seeds = np.random.randint(0, 1000, size=4)
    T = 40
    termination_by_time = False
    hparams_lims_file = "test_scripts/optimization/hparams_lims.json"
    save_dirs = ["results/pso/test/opt"+str(i) for i in range(len(seeds))]
    metric_name = "rand"

    # plotter
    plotter = Plotter(
        num_axes=len(seeds),
    )

    for i, seed in enumerate(seeds):
        print(f"optimization {i+1}/{len(seeds)}")
        rng = np.random.default_rng(seed)

        # define particle swarm and metric
        pso = ParticleSwarmOptimizationWrapper(
            hparams_lims_file=hparams_lims_file,
            save_dir=save_dirs[i],
            T=T,
            termination_by_time=termination_by_time,
            rng=rng,
        )
        metric = Metric(
            metric_name=metric_name,
            hparams_lims=pso.hparams_lims,
            rng=rng,
        )

        # run optimization
        optimize(
            pso=pso,
            metric=metric,
        ) # (N, T, M), (N, T)

        # plot results
        plotter.plot2D(
            pso=pso,
            metric=metric,
            ax_idx=i,
        )

    plotter.show()

if __name__ == "__main__":
    test_pso()