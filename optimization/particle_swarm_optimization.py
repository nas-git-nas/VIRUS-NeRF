import numpy as np
import os
import json
import time
import pandas as pd


class ParticleSwarmOptimization():
    def __init__(
        self,
        hparams_lims_file:str,
        save_dir:str,
        T_iter:int=None,
        T_time:float=None,
        rng:np.random.Generator=None,
    ):
        # set random number generator
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        # define termination condition
        self.T = 1 # number of iterations
        self.termination_by_time = False # whether to terminal condition is given by time or iterations
        if T_iter is None and T_time is None:
            print("ERROR: ParticleSwarmOptimization.__init__: Either T or T_time must be defined."
                  + " setting T to 1")
        elif T_iter is not None:
            if T_time is not None:
                print(f"WARNING: ParticleSwarmOptimization.__init__: Either T or T_time must be defined, not both."
                    + f" ignoring T_time and setting T to T_iter={T_iter}")
            self.T = T_iter
        elif T_time is not None:
            self.T = T_time
            self.termination_by_time = True
            self.start_time = time.time()

        # read hparams limits
        self.hparams_lims, self.hparams_order, self.hparams_group = self._readHparamsFile(
            hparams_lims_file=hparams_lims_file,
        ) # np.array (M, 2), dict (key: str, value: int), dict (key: str, value: str)
        self.M = self.hparams_lims.shape[0]

        # PSO parameters
        self.N = 10 # number of particles
        self.num_neighbours = 3 # number of neighbours to consider for social component
        self.alpha_momentum = 0.6
        self.alpha_propre = 0.2 * np.sqrt(self.M)
        self.alpha_social = 0.2 * np.sqrt(self.M)
        self.prob_explore = 0.75 # probability to explore instead of exploit

        # create save files
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        name_list = [param for param in self.hparams_order.keys()]
        name_list += [f"best_{param}" for param in self.hparams_order.keys()]
        name_list += ["score", "best_score", "best_count"]
        self.save_files = []
        for i in range(self.N):
            self.save_files.append(os.path.join(self.save_dir, "pso_particle_"+str(i)+".csv"))
            df = pd.DataFrame(columns=name_list)
            df.to_csv(self.save_files[-1], index=False)
            
        pso_params_dict = {
            "num_particles": [self.N],
            "num_neighbours": [self.num_neighbours], 
            "alpha_momentum": [self.alpha_momentum],
            "alpha_propre": [self.alpha_propre],
            "alpha_social": [self.alpha_social],
            "prob_explore": [self.prob_explore],
        }
        df = pd.DataFrame.from_dict(pso_params_dict)
        df.to_csv(os.path.join(self.save_dir, "pso_params.csv"), index=False)

        # position and velocity of particles
        self.pos = self._initParticlesRandom() # (N, M)
        self.vel = 0.5 * np.sqrt(self.M) * self._initParticlesRandom() # (N, M)
        self.best_pos = np.zeros_like(self.pos) # (N, M)
        self.best_score = np.full((self.N,), fill_value=np.inf) # (N,)
        self.best_count = np.zeros((self.N,), dtype=int) # (N,)

        # iteration parameters
        self.n = 0 # current particle, [0, N-1]
        self.t = 0 # current iteration, [0, T-1]
        self.exploring = True # whether to explore or exploit

    def getHparams(
        self,
        group_dict_layout:bool=False,
    ):
        """
        Get hyper parameters of current particle
        Args:
            group_dict_layout: whether to return hyper parameters as { group: { param: val } } or np.array (M,); bool
        Returns:
            hparams: hyper parameters; np.array (M,) or dict { group: { param: val } }
        """
        prob = self.rng.random()
        if (self.best_count[self.n] == 0) or (prob < self.prob_explore):
            pos = self.pos[self.n]
            self.exploring = True
        else:
            pos = self.best_pos[self.n]
            self.exploring = False

        if group_dict_layout:
            return self._pos2groupDict(
            pos=pos,
        )
        return self._pos2hparam(
            pos=pos,
        )
    
    def update(
        self,
        score:float,
    ):
        """
        Update particle swarm
        Args:
            score: score of current particle; float
        Returns:
            terminate: whether to terminate optimization or not; bool
        """
        # update best position of particle
        best_pos, best_pos_neighbourhood = self._updateBestPos(
            n=self.n,
            score=score,
        )

        # update particle
        self._updateParticle(
            n=self.n,
            best_pos=best_pos,
            best_pos_neighbourhood=best_pos_neighbourhood,
        )
        
        # update iteration parameters
        return self.updateIteration()
    
    def saveState(
        self,
        score:float,
    ):
        """
        Save state of particle swarm optimization.
        Args:
            score: score of current particle; float
        """
        # create name dictionary
        hparams = self._pos2hparam(
            pos=self.pos[self.n],
        ) # np.array (M,)
        name_dict = self._hparam2nameDict(
            hparams=hparams,
        ) # dict (key: str, value: float)
        best_hparams = self._pos2hparam(
            pos=self.best_pos[self.n],
        ) # np.array (M,)
        best_dict = self._hparam2nameDict(
            hparams=best_hparams,
        ) # dict (key: str, value: float)
        best_name_dict = {f"best_{param}": val for param, val in best_dict.items()}
        name_dict.update(best_name_dict)
        name_dict["score"] = score
        name_dict["best_score"] = self.best_score[self.n]
        name_dict["best_count"] = self.best_count[self.n]

        # save updated csv file
        df = pd.read_csv(self.save_files[self.n])
        df = pd.concat([df, pd.DataFrame(name_dict, index=[0])], axis=0, ignore_index=True)
        df.to_csv(self.save_files[self.n], index=False)
    
    def _readHparamsFile(
        self,
        hparams_lims_file:str,
    ):
        """
        Read hyper parameter limits from json file
        Args:
            hparams_lims_file: file path of json file; str
        Returns:
            hparams_lims: hyper parameters limits; np.array (M, 2)
            hparams_order: hyper parameters order; dict (key: str, value: int)
            hparams_group: hyper parameters group; dict (key: str, value: str)
        """
        with open(hparams_lims_file) as f:
            group_dict = json.load(f)

        name_dict, hparams_group = self._groupDict2nameDict(
            group_dict=group_dict,
            return_groups=True,
        )

        hparams_order = {}
        hparams_lims = []
        for i, (param, lims) in enumerate(name_dict.items()):
            hparams_order[param] = i
            hparams_lims.append(lims)
        hparams_lims = np.array(hparams_lims)
        
        return hparams_lims, hparams_order, hparams_group
    
    def _initParticlesRandom(
        self,
    ):
        """
        Initialize particles randomly.
        Returns:
            pos: particle space; np.array (N, M)
        """
        return self.rng.random(size=(self.N, self.M))
    
    def _updateBestPos(
        self,
        n:int,
        score:float,
    ):
        """
        Update best position of particle.
        Args:
            n: index of particle; int
            score: score of particle; float
        Returns:
            best_pos: best position of particle n; np.array (M,)
            best_pos_neighbourhood: best position in neighbourhood; np.array (M,)
        """
        # update best score of particle in case of exploration or exploitation
        if self.exploring: 
            if score < self.best_score[n]:
                self.best_score[n] = score
                self.best_pos[n] = self.pos[n]
                self.best_count[n] = 1
        else:
            self.best_score[n] = (self.best_count[n]*self.best_score[n] + score) / (self.best_count[n] + 1)
            self.best_count[n] += 1

        # determine best particle
        dists = np.sum((self.pos - self.pos[n])**2, axis=-1) # (N,)
        neighbours = np.argsort(dists)[:self.num_neighbours+1] # (num_neighbours,)
        best_neighbour = np.argmin(self.best_score[neighbours])
        best_pos_neighbourhood = self.best_pos[neighbours[best_neighbour]]

        return self.best_pos[n], best_pos_neighbourhood
    
    def _updateParticle(
        self,
        n:int,
        best_pos:np.array,
        best_pos_neighbourhood:np.array,
    ):
        """
        Update particle.
        Args:
            n: index of particle; int
            best_pos: best position of particle n; np.array (M,)
            best_pos_neighbourhood: best position in neighbourhood; np.array (M,)
        """
        # update velocity and position of current particle
        self.vel[n] = self.alpha_momentum * self.vel[n] \
            + self.alpha_propre * self.rng.random() * (best_pos - self.pos[n]) \
            + self.alpha_social * self.rng.random() * (best_pos_neighbourhood - self.pos[n])
        self.pos[n] = np.clip(self.pos[n] + self.vel[n], 0, 1)

    def updateIteration(
        self,
    ):
        """
        Update iteration parameters.
        Returns:
            terminate: whether to terminate optimization; bool
        """
        if self.n == self.N - 1:
            self.n = 0
            self.t += 1
        else:
            self.n += 1

        # check termination condition
        if self.termination_by_time:
            if (time.time() - self.start_time) > self.T and (self.n == 0):
                return True
        else:
            if self.t == self.T:
                return True
        return False
    
    def _groupDict2pos(
        self,
        group_dict:dict,
    ):
        """
        Convert hyper parameters { group: { param: val } } to position in particle space.
        Args:
            group_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        Returns:
            pos: particle space; np.array (M,)
        """
        name_dict = self._groupDict2nameDict(
            group_dict=group_dict,
        )
        hparams = self._nameDict2hparam(
            name_dict=name_dict,
        )
        pos = self._hparam2pos(
            hparams=hparams,
        )
        return pos
    
    def _pos2groupDict(
        self,
        pos:np.array,
    ):
        """
        Convert position in particle space to hyper parameters { group: { param: val } }.
        Args:
            pos: particle space; np.array (M,)
        Returns:
            group_dict: dictionary containing hyper parameters; dict (key: str, value: str)
        """
        hparams = self._pos2hparam(
            pos=pos,
        )
        name_dict = self._hparam2nameDict(
            hparams=hparams,
        )
        group_dict = self._nameDict2groupDict(
            name_dict=name_dict,
        )
        return group_dict
    
    def _hparam2pos(
        self,
        hparams:np.array,
    ):
        """
        Convert hyper parameters to particle space.
        Args:
            hparams: hyper parameters; np.array (M,) or (N, M)
        Returns:
            pos: particle space; np.array (M,) or (N, M)
        """
        return (hparams - self.hparams_lims[:,0]) / (self.hparams_lims[:,1] - self.hparams_lims[:,0])
    
    def _pos2hparam(
        self,
        pos:np.array,
    ):
        """
        Convert particle space to hyper parameters.
        Args:
            pos: particle space; np.array (M,) or (N, M)
        Returns:
            hparams: hyper parameters; np.array (M,) or (N, M)
        """
        return pos * (self.hparams_lims[:,1] - self.hparams_lims[:,0]) + self.hparams_lims[:,0]
    
    def _nameDict2hparam(
        self,
        name_dict:dict,
    ):
        """
        Convert hyper-parameter dictionary from { param: val } to np.array (M,).
        Args:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        Returns:
            hparams: hyper parameters; np.array (M,)
        """
        hparams = np.zeros(self.M)
        for param, i in self.hparams_order.items():
            hparams[i] = name_dict[param]
        return hparams
    
    def _hparam2nameDict(
        self,
        hparams:np.array,
    ):
        """
        Convert hyper-parameter dictionary from np.array (M,) to { param: val }.
        Args:
            hparams: hyper parameters; np.array (M,)   
        Returns:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        """
        name_dict = {}
        for param, i in self.hparams_order.items():
            name_dict[param] = hparams[i]
        return name_dict
    
    def _nameDict2groupDict(
        self,
        name_dict:dict,
    ):
        """
        Convert hyper-parameter dictionary from
            { param: val } to { group: { param: val } }
        Args:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        Returns:
            group_dict: dictionary containing hyper parameters; dict (key: str, value: str)
        """
        group_dict = { group: {} for group in np.unique(list(self.hparams_group.values())) }
        for param, val in name_dict.items():
            group_dict[self.hparams_group[param]][param] = val
        return group_dict
    
    def _groupDict2nameDict(
        self,
        group_dict:dict,
        return_groups:bool=False,
    ):
        """
        Convert hyper-parameter dictionary from
            { group: { param: val } } to { param: val }
        Args:
            group_dict: dictionary containing hyper parameters; dict (key: str, value: str)
            return_groups: whether to return groups or not; bool
        Returns:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        """
        name_dict = {}
        groups = {}
        for group, group_params in group_dict.items():
            for param, val in group_params.items():
                if param in name_dict:
                    print(f"ERROR: ParticleSwarmOptimization._group2nameHparams: parameter {param} is defined multiple times.")
                name_dict[param] = val   
                groups[param] = group

        if return_groups:
            return name_dict, groups
        return name_dict