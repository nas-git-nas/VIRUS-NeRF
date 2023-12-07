import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from optimization.particle_swarm_optimization_wrapper import ParticleSwarmOptimizationWrapper
from optimization.metric import Metric



class Cmaps():
    def __init__(
        self,
        num_cmaps:int,
        norm_min:float,
        norm_max:float,
        skip_bright_colors:bool=False,
    ) -> None:
        cmap_names = ["Blues", "Greens", "Purples", "Oranges", "Reds"]
        self.cmaps = [matplotlib.colormaps[cmap_names[n%len(cmap_names)]] for n in range(num_cmaps)]

        self.skip_bright_colors = skip_bright_colors
        if self.skip_bright_colors:
            self.norm_delta = (norm_max - norm_min) / 2
        self.norm = matplotlib.colors.Normalize(vmin=norm_min, vmax=norm_max + self.norm_delta)

    def __call__(
        self,
        cmap_idx:int,
        val:float,
    ):
        """
        Determine color based on cmap and value.
        Args:
            cmap_idx: index of colormap; int
            val: value to determine color; float
        Returns:
            color: color; np.array (4,)
        """
        if self.skip_bright_colors:
            val += self.norm_delta
        return self.cmaps[cmap_idx](self.norm(val))
    

class Plotter():
    def __init__(
        self,
        num_axes:int,
    ) -> None:
        # create figure
        self.num_rows = np.ceil(np.sqrt(num_axes)).astype(int)
        self.num_cols = self.num_rows
        self.fig, self.axes = plt.subplots(ncols=self.num_cols, nrows=self.num_rows, figsize=(9,7))

    def show(
        self,
    ):
        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.1, 0.05, 0.8]) # [left, bottom, width, height]
        self.fig.colorbar(self.im, cax=cbar_ax)
        plt.show()
    
    def plot2D(
        self,
        pso:ParticleSwarmOptimizationWrapper,
        metric:Metric,
        ax_idx:int,
        res:int=64,
    ):
        if isinstance(ax_idx, matplotlib.axes.Axes):
            ax = self.axes
        else:
            axes_i, axes_j = np.unravel_index(ax_idx, self.axes.shape)
            ax = self.axes[axes_i, axes_j]

        pos, vel, best_pos, best_score = self._loadData(
            pso=pso,
        ) # (N, M, L), (N, M, L), (N, M, L)
        N = pos.shape[0] # number of particles
        M = pos.shape[1] # number of hparams
        L = pos.shape[2] # number of iterations

        # interfere gaussian
        m1, m2 = np.meshgrid(
            np.linspace(pso.hparams_lims[0, 0], pso.hparams_lims[0, 1], num=res),
            np.linspace(pso.hparams_lims[1, 0], pso.hparams_lims[1, 1], num=res),
            indexing='ij',
        )
        X = np.stack((m1.flatten(), m2.flatten()), axis=-1)
        scores = metric(
            X=X,
        )
        scores = scores.reshape((res, res))

        extent = [pso.hparams_lims[0, 0], pso.hparams_lims[0, 1], pso.hparams_lims[1,0], pso.hparams_lims[1, 1]]
        self.im = ax.imshow(scores.T, origin='lower', extent=extent, cmap='Greys', vmin=0, vmax=1)

        cmaps = Cmaps(
            num_cmaps=N,
            norm_min=0,
            norm_max=M-2,
            skip_bright_colors=True,
        )
        for n in range(N):
            for l in range(L-1):
                ax.plot([pos[n, 0, l], pos[n, 0, l+1]], [pos[n, 1, l], pos[n, 1, l+1]], 
                        color=cmaps(n, l), linewidth=2)

        ax.scatter(metric.centre[0], metric.centre[1], color="black", s=200, marker='*') 
        for n in range(N):
            ax.scatter(best_pos[n, 0, -1], best_pos[n, 1, -1], color=cmaps(n, L-2), s=100, marker='*') 
            ax.scatter(pos[n, 0, 0], pos[n, 1, 0], color=cmaps(n, 0), s=10) 

        hparams_order_inv = {}
        for hparam in pso.hparams_order.keys():
            if pso.hparams_order[hparam] in hparams_order_inv.keys():
                print("ERROR: test_psoGauss: more than one parameter with order 0")
            hparams_order_inv[pso.hparams_order[hparam]] = hparam
        if axes_i == self.num_rows-1:
            ax.set_xlabel(str(hparams_order_inv[0]))
        else:
            ax.set_xticks([])   
        if axes_j == 0:
            ax.set_ylabel(str(hparams_order_inv[1]))
        else:
            ax.set_yticks([])

        best_idx = np.argmin(best_score[:,-1])
        ax.set_title(f"score={best_score[best_idx,-1]:.3f}, "
                    + f"dist={np.linalg.norm(metric.centre - best_pos[best_idx,:,-1]):.2f}")

        self.axes[axes_i, axes_j] = ax

    def _loadData(
        self,
        pso:ParticleSwarmOptimizationWrapper,
    ):
        """
        Load data from files.
        Args:
            pso: particle swarm optimization; ParticleSwarmOptimizationWrapper
        Returns:
            pos: position of particles; np.array (N, M, L)
            vel: velocity of particles; np.array (N, M, L)
            best_pos: best position of particles; np.array (N, M, L)
            best_score: best score of particles; np.array (N, L)
        """
        N = pso.N # number of particles
        L = pso.T // pso.N # number of iterations per particle
        M = pso.M # number of hyperparameters

        pos = np.zeros((N, M, L))
        vel = np.zeros((N, M, L))
        best_pos = np.zeros((N, M, L))
        best_score = np.zeros((N, L))
        for i in range(N):
            # load data
            pos_dict = pso._loadStateFromFile(
                file_path=pso.pos_files[i],
                return_last_row=False,
            ) # dict of lists of floats
            del pos_dict["score"]
            del pos_dict["time"]
            del pos_dict["iteration"]     

            best_pos_dict = pso._loadStateFromFile(
                file_path=pso.best_pos_files[i],
                return_last_row=False,
            ) # dict of lists of floats
            best_score[i] = np.array(best_pos_dict["best_score"])
            del best_pos_dict["best_score"]
            del best_pos_dict["best_count"]

            vel_dict = pso._loadStateFromFile(
                file_path=pso.vel_files[i],
                return_last_row=False,
            ) # dict of lists of floats

            # convert to np.array
            pos[i] = pso._nameDict2hparam(
                name_dict=pos_dict,
            ) # np.array (L, M)

            vel[i] = pso._nameDict2hparam(
                name_dict=vel_dict,
            ) # np.array (L, M)

            best_pos[i] = pso._nameDict2hparam(
                name_dict=best_pos_dict,
            ) # np.array (L, M)

        return pos, vel, best_pos, best_score
