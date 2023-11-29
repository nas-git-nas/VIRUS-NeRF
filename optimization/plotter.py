import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from optimization.particle_swarm_optimization import ParticleSwarmOptimization
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
        pso:ParticleSwarmOptimization,
        metric:Metric,
        X_list:list,
        score_list:list,
        ax_idx:int,
        res:int=64,
    ):
        axes_i, axes_j = np.unravel_index(ax_idx, self.axes.shape)
        ax = self.axes[axes_i, axes_j]

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

        X_list = np.array(X_list) # (N, T, M)
        cmaps = Cmaps(
            num_cmaps=X_list.shape[0],
            norm_min=0,
            norm_max=X_list.shape[1]-2,
            skip_bright_colors=True,
        )
        for n in range(X_list.shape[0]):
            for t in range(X_list.shape[1]-1):
                ax.plot([X_list[n, t, 0], X_list[n, t+1, 0]], [X_list[n, t, 1], X_list[n, t+1, 1]], 
                        color=cmaps(n, t), linewidth=2)

        score_list = np.array(score_list) # (N, T)
        ax.scatter(metric.centre[0], metric.centre[1], color="black", s=200, marker='*') 
        for n in range(X_list.shape[0]):
            best_idx = np.argmin(score_list[n])
            ax.scatter(X_list[n, best_idx, 0], X_list[n, best_idx, 1], color=cmaps(n, X_list.shape[1]-2), s=100, marker='*') 
            ax.scatter(X_list[n, 0, 0], X_list[n, 0, 1], color=cmaps(n, 0), s=10) 

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

        best_idxs = np.unravel_index(np.argmin(score_list), score_list.shape)
        ax.set_title(f"score={score_list[best_idxs[0], best_idxs[1]]:.2f}, "
                    + f"dist={np.linalg.norm(metric.centre - X_list[best_idxs[0], best_idxs[1]]):.2f}")

        self.axes[axes_i, axes_j] = ax