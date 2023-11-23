import numpy as np
import os
import sys
 
sys.path.insert(0, os.getcwd())
from optimization.particle_swarm_optimization import ParticleSwarmOptimization


def metricGauss(
    hparams:dict,
    hparams_order:dict,
    centre:np.ndarray,
    std:np.ndarray,
):
    """
    Evaluate metric.
    Args:
        hparams: hyper parameters; dict (key: str, value: float)
        hparams_order: order of hparams; dict (key: str, value: int)
        center: center of gaussian; np.array (M,)
        std: standard deviation of gaussian; np.array (M,)
    Returns:
        score: score of input; float
        X: position in hparams space; np.array (M,)
    """
    # convert hparams to position in hparams space
    X = np.zeros((len(hparams.keys()),))
    for key, i in hparams_order.items():
        X[i] = hparams[key]

    # evaluate score
    score_inv = np.exp(- (X-centre)@np.diag(1/std**2)@(X-centre))
    score = 1 - score_inv
    return score, X

def test_psoGauss():
    # define optimization algorithm
    seed = 29
    pso = ParticleSwarmOptimization(
        N=2,
        hparams_lims_file="test_scripts/optimization/hparams_lims.json",
        T_iter=10,
        T_time=None,
        seed=seed,
    )

    # define metric
    centre = np.zeros((pso.M,))
    for key, i in pso.hparams_order.items():
        centre[i] = np.random.uniform(pso.hparams_lims[key][0], pso.hparams_lims[key][1], seed=seed)
    std = np.random.uniform(0, 1, size=(pso.M,))

    # run optimization
    terminate = False
    X_list = []
    score_list = []
    while not terminate:
        # get hparams to evaluate
        hparams = pso.getHparams()

        # evaluate metric
        score, X = metricGauss(
            hparams=hparams,
            hparams_order=pso.hparams_order,
            centre=centre,
            std=std,
        )
        X_list.append(X)
        score_list.append(score)

        # update particle swarm
        terminate = pso.update(
            score=score,
        )



if __name__ == "__main__":
    test_psoGauss()