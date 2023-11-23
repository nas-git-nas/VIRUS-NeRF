import numpy as np
import os
import json


class ParticleSwarmOptimization():
    def __init__(
        self,
        T:int,
        N:int,
        hparams_json_file:str,
        seed:int=29,
    ):
        self.T = T # number of iterations
        self.N = N # number of particles
        self.M = 0 # number of optimization dimensions
        self.rng = np.random.default_rng(
            seed=seed,
        ) # random number generator

        # read hparams limits
        self.hparams_lims, self.hparams_order, self.M = self._readHparamsFile(
            hparams_json_file=hparams_json_file,
        )

        # parameters for PSO
        self.alpha_momentum = 0.6
        self.alpha_propre = 0.2
        self.alpha_social = 0.2

        # position and velocity of particles
        self.pos = self._initParticlesRandom() # (N, M)
        self.vel = self._initParticlesRandom() # (N, M)
        self.best_pos = np.zeros_like(self.pos) # (N, M)
        self.best_score = np.full((self.N,), fill_value=np.inf) # (N,)

        # iteration parameters
        self.n = 0 # current particle, [0, N-1]
        self.t = 0 # current iteration, [0, T-1]


    def getHparams(
        self,
    ):
        """
        Get hyper parameters of current particle
        Returns:
            hparams: hyper parameters; dict
        """
        return self._particleSpace2Hparams(
            p=self.pos[self.n],
        )
    
    def update(
        self,
        score:float,
    ):
        """
        Update particle swarm
        Args:
            score: score of current particle; float
        """
        # update best score of particle
        if score < self.best_score[self.n]:
            self.best_score[self.n] = score
            self.best_pos[self.n] = self.pos[self.n]

        # determine best particle
        best_idx = np.argmin(self.best_score)
        best_pos_total = self.best_pos[best_idx]

        # update velocity and position of current particle
        self.vel[self.n] = self.alpha_momentum * self.vel[self.n] \
            + self.alpha_propre * self.rng.random() * (self.best_pos[self.n] - self.pos[self.n]) \
            + self.alpha_social * self.rng.random() * (best_pos_total - self.pos[self.n])
        self.pos[self.n] = np.clip(self.pos[self.n] + self.vel[self.n], 0, 1)
        


    def _initParticlesRandom(
        self,
    ):
        return self.rng.random(size=(self.N, self.M))
    
    def _readHparamsFile(
        self,
        hparams_json_file:str,
    ):
        file_path = os.path.join("optimization", hparams_json_file)
        with open(file_path) as f:
            hparams_lims = json.load(f)

        hparams_order = {}
        for i, key in enumerate(hparams_lims.keys()):
            hparams_order[key] = i

        M = len(hparams_lims.keys())
        return hparams_lims, hparams_order, M
    
    def _hparams2ParticleSpace(
        self,
        hparams:dict,
    ):
        """
        Convert hyper parameters to particle space.
        Args:
            hparams: dictionary containing hyper parameters; dict (key: str, value: float)
        Returns:
            pos: particle space; np.array (M,)
        """
        pos = np.zeros(self.M)
        for key, i in self.hparams_order.items():
            pos[i] = (hparams[key] - self.hparams_lims[key][0]) / (self.hparams_lims[key][1] - self.hparams_lims[key][0])
        return pos
    
    def _particleSpace2Hparams(
        self,
        pos:np.array,
    ):
        """
        Convert particle space to hyper parameters.
        Args:
            pos: particle space; np.array (M,)
        Returns:
            hparams: dictionary containing hyper parameters; dict (key: str, value: float)
        """
        hparams = {}
        for key, i in self.hparams_order.items():
            hparams[key] = pos[i] * (self.hparams_lims[key][1] - self.hparams_lims[key][0]) + self.hparams_lims[key][0]
        return hparams