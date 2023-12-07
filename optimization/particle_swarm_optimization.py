import numpy as np


class ParticleSwarmOptimization():
    def __init__(
        self,
        rng:np.random.Generator,
        pso_params_dict:dict,
        pso_init_dict:dict=None,
        current_particle:int=0,
    ):
        """
        Initialize particle swarm optimization.
        Args:
            rng: random number generator; np.random.Generator
            pso_params_dict: dictionary of PSO parameters; dict
            pso_init_dict: dictionary of PSO initialization; dict
            current_particle: current particle; int
        """
        # general
        self.rng = rng

        # PSO parameters
        self.n = int(current_particle) # current particle, [0, N-1]
        self.N = pso_params_dict["num_particles"] # number of particles
        self.M = pso_params_dict["num_dimensions"] # number of dimensions
        self.num_neighbours = pso_params_dict["num_neighbours"] # number of neighbours
        self.alpha_momentum = pso_params_dict["alpha_momentum"] # momentum coefficient
        self.alpha_propre = pso_params_dict["alpha_propre"] # propre coefficient
        self.alpha_social = pso_params_dict["alpha_social"] # social coefficient
        self.prob_explore = pso_params_dict["prob_explore"] # probability of exploration
        self.exploring = True # whether to explore or exploit

        # initialize position and velocity of particles
        if pso_init_dict is None:
            self.pos, self.vel, self.best_pos, self.best_score, self.best_count = self._initParticles()
        else:
            self.pos = pso_init_dict["pos"]
            self.vel = pso_init_dict["vel"]
            self.best_pos = pso_init_dict["best_pos"]
            self.best_score = pso_init_dict["best_score"]
            self.best_count = pso_init_dict["best_count"]

    def nextPos(
        self,
    ):
        """
        Get next position of particle.
        Returns:
            pos: position of particle; np.array (M,)
        """
        prob = self.rng.random()
        if (self.best_count[self.n] == 0) or (prob < self.prob_explore):
            self.exploring = True
            return self.pos[self.n]
  
        self.exploring = False
        return self.best_pos[self.n]
    
    def updatePSO(
        self,
        score:float,
    ):
        """
        Update particle using score of current position.
        Args:
            score: score of current particle; float
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
        self._updateIteration()

    def _initParticles(
        self,
    ):
        """
        Initialize particles.
        Returns:
            pos: particle space; np.array (N, M)
            vel: particle velocity; np.array (N, M)
            best_pos: best position of particle; np.array (N, M)
            best_score: best score of particle; np.array (N,)
            best_count: number of times best score was updated; np.array (N,)
        """
        pos = self._initParticlesRandom() # (N, M)
        vel = 0.5 * np.sqrt(self.M) * self._initParticlesRandom() # (N, M)
        best_pos = np.zeros_like(pos) # (N, M)
        best_score = np.full((self.N,), fill_value=np.inf) # (N,)
        best_count = np.zeros((self.N,), dtype=int) # (N,)
        return pos, vel, best_pos, best_score, best_count
    
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

    def _updateIteration(
        self,
    ):
        """
        Update iteration parameters.
        Returns:
            terminate: whether to terminate optimization; bool
        """
        if self.n == self.N - 1:
            self.n = 0
        else:
            self.n += 1
    
