import numpy as np


class Metric():
    def __init__(
        self,
        metric_name:str,
        hparams_lims:np.ndarray,
        rng:np.random.Generator,
    ) -> None:
        """
        Args:
            metric_name: name of metric; str
            hparams_lims: limits of hparams; np.array (M, 2)
            rng: random number generator; np.random.Generator
        """
        self.name = metric_name
        self.rng = rng

        # set parameters
        delta = hparams_lims[:, 1]-hparams_lims[:, 0]
        self.centre = self.rng.uniform(hparams_lims[:, 0], hparams_lims[:, 1])
        self.std = self.rng.uniform(delta/6, delta/3)
        self.freq = self.rng.uniform(delta/6, delta)
        self.rand_std = 0.1 #self.rng.uniform(0.1, 0.8)

    def __call__(
        self,
        X:np.ndarray,
    ):
        if self.name == "gauss":
            return self.gauss(
                X=X,
            ) # float  
        elif self.name == "cos":
            return self.cos(
                X=X,
            ) # float
        elif self.name == "rand":
            return self.rand(
                X=X,
            ) # float
        else:
            print(f"ERROR: Metric.__call__: metric_name {self.name} not supported")

    def gauss(
        self,
        X:np.ndarray,
    ):
        """
        Evaluate gaussian.
        Args:
            X: position in hparams space; np.array (M,) or (N, M)
        Returns:
            score: score of input; float or np.array (N,)
        """
        score_inv = np.exp(- np.sum((X-self.centre)**2 / self.std**2, axis=-1))
        score = 1 - score_inv
        return score

    def cos(
        self,
        X:np.ndarray,
    ):
        """
        Evaluate cosine-gaussian.
        Args:
            X: position in hparams space; np.array (M,) or (N, M)
        Returns:
            score: score of input; float or np.array (N,)
        """
        exp_score = self.gauss(
            X=X,
        ) # float
        cos_score_inv = np.prod((np.cos(2*np.pi * self.freq * (X-self.centre))+1)/2, axis=-1)
        cos_score = 1 - cos_score_inv
        return cos_score * exp_score
    
    def rand(
        self,
        X:np.ndarray,
    ):
        """
        Evaluate random metric.
        Args:
            X: position in hparams space; np.array (M,) or (N, M)
        Returns:
            score: score of input; float or np.array (N,)
        """
        score = self.cos(
            X=X,
        ) # float
        rand_score = self.rng.normal(0, self.rand_std)
        return np.clip(score + rand_score, 0, 1)