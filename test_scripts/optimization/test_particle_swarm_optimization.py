import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from abc import ABC, abstractmethod
 
sys.path.insert(0, os.getcwd())
from optimization.particle_swarm_optimization import ParticleSwarmOptimization
from optimization.metric import Metric
from optimization.plotter import Plotter


def optimize(
    pso:ParticleSwarmOptimization,
    metric:Metric,
):
    """
    Run optimization.
        N: number of particles; int
        M: number of hparams; int
        T: number of iterations; int
    Args:
        pso: particle swarm optimization; ParticleSwarmOptimization
        metric: metric to optimize; Metric
    Returns:
        X_list: list of hparams; np.array (N, T, M)
        score_list: list of scores; np.array (N, T)
    """
    terminate = False
    X_list = [ [] for _ in range(pso.N)] # (N, T, M)
    score_list = [ [] for _ in range(pso.N)] # (N, T)
    while not terminate:
        # get hparams to evaluate
        X = pso.getHparams(
            group_dict_layout=False,
        ) # np.array (M,)

        # evaluate metric
        score = metric(
            X=X,
        ) # float

        # save state
        pso.saveState(
            score=score,
        )
        X_list[pso.n].append(X)
        score_list[pso.n].append(score)

        # update particle swarm
        terminate = pso.update(
            score=score,
        ) # bool

    return np.array(X_list), np.array(score_list)


def test_pso():
    # define optimization algorithm
    seeds = np.random.randint(0, 1000, size=9)
    T_iter = 30
    T_time = None
    hparams_lims_file = "test_scripts/optimization/hparams_lims.json"
    save_dir = "results/pso/test"
    metric_name = "rand"

    # plotter
    plotter = Plotter(
        num_axes=len(seeds),
    )

    for i, seed in enumerate(seeds):
        print(f"optimization {i+1}/{len(seeds)}")
        rng = np.random.default_rng(seed)

        # define particle swarm and metric
        pso = ParticleSwarmOptimization(
            hparams_lims_file=hparams_lims_file,
            save_dir=save_dir,
            T_iter=T_iter,
            T_time=T_time,
            rng=rng,
        )
        metric = Metric(
            metric_name=metric_name,
            hparams_lims=pso.hparams_lims,
            rng=rng,
        )

        # run optimization
        X_list, score_list = optimize(
            pso=pso,
            metric=metric,
        ) # (N, T, M), (N, T)

        # plot results
        plotter.plot2D(
            pso=pso,
            metric=metric,
            X_list=X_list,
            score_list=score_list,
            ax_idx=i,
        )

    plotter.show()

if __name__ == "__main__":
    test_pso()