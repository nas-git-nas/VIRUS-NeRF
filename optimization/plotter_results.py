import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import json


class PlotterResults():
    def __init__(
            self,
            data_dir,
        ) -> None:
        self.data_dir = data_dir

        self.score_min = 0.15
        self.score_max = 0.3
        self.num_particles = 8
        self.keep_best_n = 5
        self.best_symbs = ['*', 'o', 'v', 'x', '+', '^', '<', '>', 's', 'p', 'P', 'h', 'H', 'X', 'D', 'd', '|', '_']
        self.best_symbs = self.best_symbs[:self.keep_best_n]

    def plot(
            self,
        ):
        """
        Plot results from data_dir.
        """
        # read data
        pos, scores, parameters = self._readPosData()
        vel, parameters_vel = self._readPosData(
            read_vel=True
        )
        best_pos, best_scores, best_iters, best_parameters = self._readBestPosData(
            keep_best_n_particles=self.keep_best_n,
        )
        hparams_lims = self._readHparamsLims()

        # verify that parameters are the same
        assert parameters == best_parameters
        assert parameters == parameters_vel

        # keep only best N scores
        idxs = np.argsort(best_scores)
        best_pos = best_pos[idxs[:self.keep_best_n], :]
        best_scores = best_scores[idxs[:self.keep_best_n]]

        # reverse colotmap
        cmap = matplotlib.colormaps['jet']
        cmap_inv = cmap.reversed() 


        # plot
        fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(9,10))

        axes[0] = self._plotParticleSpeeds(
            vel=vel,
            scores=scores,
            best_iters=best_iters,
            best_scores=best_scores,
            hparams_lims=hparams_lims,
            parameters=parameters,
            ax=axes[0],
            cmap_inv=cmap_inv,
        )

        axes[1] = self._plotParticleScores(
            scores=scores,
            best_scores=best_scores,
            best_iters=best_iters,
            ax=axes[1],
            cmap_inv=cmap_inv,
        )

        axes[2], im = self._plotHparams(
            pos=pos,
            scores=scores,
            best_pos=best_pos,
            best_scores=best_scores,
            best_iters=best_iters,
            hparams_lims=hparams_lims,
            parameters=parameters,
            ax=axes[2],
            cmap_inv=cmap_inv,
        )

        # add colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.05, 0.8]) # [left, bottom, width, height]
        # cbar_ax.set_title('Mean NND',fontsize=13)
        fig.colorbar(im, cax=cbar_ax)

        axes[0].set_title('Particle Swarm Optimization')

        # save figure
        fig.savefig(os.path.join(self.data_dir, 'pso_results.png'))
        fig.savefig(os.path.join(self.data_dir, 'pso_results.pdf'))
        plt.show()

    def _plotParticleSpeeds(
        self,
        vel:np.ndarray,
        scores:np.ndarray,
        best_scores:np.ndarray,
        best_iters:np.ndarray,
        hparams_lims:dict,
        parameters:dict,
        ax:matplotlib.axes.Axes,
        cmap_inv:matplotlib.colors.LinearSegmentedColormap,
    ):
        # normalize velocities
        for i, param in parameters.items():
            vel[:, :, i] = (vel[:, :, i] - hparams_lims[param][0]) / (hparams_lims[param][1] - hparams_lims[param][0])

        vel_norm = np.linalg.norm(vel, axis=2) # (N, T)

        for i in range(self.keep_best_n):
            ax.scatter(np.arange(vel.shape[1]), vel_norm[i], c=scores[i], 
                           cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[i])
            ax.scatter(np.arange(vel.shape[1])[best_iters[i]], vel_norm[i, best_iters[i]], c=best_scores[i], 
                           cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[i], s=200)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Particle Speed')

        return ax
    
    def _plotParticleScores(
        self,
        scores:np.ndarray,
        best_scores:np.ndarray,
        best_iters:np.ndarray,
        ax:matplotlib.axes.Axes,
        cmap_inv:matplotlib.colors.LinearSegmentedColormap,
    ):
        for i in range(self.keep_best_n):
            ax.scatter(np.arange(scores.shape[1]), scores[i], c=scores[i], 
                           cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[i])
            ax.scatter(np.arange(scores.shape[1])[best_iters[i]], best_scores[i], c=best_scores[i], 
                           cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[i], s=200)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean NND')

        return ax
        
    def _plotHparams(
        self,
        pos:np.ndarray,
        scores:np.ndarray,
        best_pos:np.ndarray,
        best_scores:np.ndarray,
        best_iters:np.ndarray,
        hparams_lims:dict,
        parameters:dict,
        ax:matplotlib.axes.Axes,
        cmap_inv:matplotlib.colors.LinearSegmentedColormap,
    ):
        column_width = 0.35
        N = pos.shape[0]
        T = pos.shape[1]

        # normalize positions
        for i, param in parameters.items():
            pos[:, :, i] = (pos[:, :, i] - hparams_lims[param][0]) / (hparams_lims[param][1] - hparams_lims[param][0])
            best_pos[:, i] = (best_pos[:, i] - hparams_lims[param][0]) / (hparams_lims[param][1] - hparams_lims[param][0])

        for i, param in parameters.items():

            x_axis = i + column_width * np.linspace(-0.5, 0.5, T) # (T,)
            x_axis_tile = np.tile(x_axis, (N, 1)) # (N, T)

            im = ax.scatter(x_axis_tile.flatten(), pos[:, :, i].flatten(), c=scores.flatten(), 
                            cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max)
            
            for j in range(best_pos.shape[0]):
                ax.scatter(x_axis[best_iters[j]], best_pos[j, i], c=best_scores[j], 
                           cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[j], s=200)

        ax.set_xticks(list(parameters.keys()))
        ax.set_xticklabels([param.replace('_', ' ') + f":\n     [{hparams_lims[param][0]:.2f}, {hparams_lims[param][1]:.2f}]" 
                            for param in parameters.values()], rotation=20, fontsize=9)
        ax.set_ylabel('Normalized Hyper-Parameters')
        return ax, im

    def _readPosData(
            self,
            read_vel=False,
        ):
        """
        Read position data from data_dir.
        Args:
            read_vel: read velocity instead of position; bool
        Returns:
            pos: particle positions; numpy array of shape (N, T, M)
            scores: particle scores; numpy array of shape (N, T)
            parameters: dictionary of parameters { column_index: parameter_name}; dictionary { int: str }
        """
        # read parameters and number of iterations
        if read_vel:
            files = [f'pso_vel_{i}.csv' for i in range(self.num_particles)]
        else:
            files = [f'pso_pos_{i}.csv' for i in range(self.num_particles)]
        df = pd.read_csv(os.path.join(self.data_dir, files[0]))
        columns = df.columns.to_list()
        if not read_vel:
            columns.remove('score')
            columns.remove('time')
            columns.remove('iteration')
        parameters = { i:param for i, param in enumerate(columns) }

        # read data
        pos = np.full((self.num_particles, len(df), len(columns)), np.nan)
        if not read_vel:
            scores = np.full((self.num_particles, len(df)), np.nan)

        for i, file in enumerate(files):
            df = pd.read_csv(os.path.join(self.data_dir, file))
            pos_temp = df[columns].to_numpy()
            pos[i, :len(pos_temp), :] = pos_temp

            if not read_vel:
                scores_temp = df[['score']].to_numpy()
                scores[i, :len(scores_temp)] = scores_temp.flatten()

        # verify that each column has at maximum one nan
        nans = np.sum(np.isnan(pos), axis=1)
        assert np.all(nans <= 1), f"Each column should have at maximum one nan, but found {nans}."

        if not read_vel:
            return pos, scores, parameters
        else:
            return pos, parameters
    
    def _readBestPosData(
            self,
            keep_best_n_particles,
        ):
        """
        Read position data from data_dir.
        Args:
            keep_best_n_particles: keep only best N particles; int
        Returns:
            pos: particle positions; numpy array of shape (n, M)
            scores: particle scores; numpy array of shape (n,)
            parameters: dictionary of parameters { column_index: parameter_name}; dictionary { int: str }
        """
        # read parameters and number of iterations
        files = [f'pso_best_pos_{i}.csv' for i in range(self.num_particles)]
        df = pd.read_csv(os.path.join(self.data_dir, files[0]))
        columns = df.columns.to_list()
        columns.remove('best_score')
        columns.remove('best_count')
        parameters = { i:param for i, param in enumerate(columns) }

        # read data
        best_pos = np.zeros((len(files), len(columns)))
        best_scores = np.zeros((len(files)))
        best_iters = np.zeros((len(files)), dtype=int)
        for i, file in enumerate(files):
            df = pd.read_csv(os.path.join(self.data_dir, file))
            pos_temp = df[columns].to_numpy()
            scores_temp = df[['best_score']].to_numpy().flatten()

            best_pos[i] = pos_temp[-1, :]
            best_scores[i] = scores_temp[-1]
            best_iters[i] = np.argmax(scores_temp == best_scores[i])

        # keep only best n particles
        idxs = np.argsort(best_scores)
        idxs = idxs[:keep_best_n_particles]
        best_pos = best_pos[idxs, :]
        best_scores = best_scores[idxs]

        return best_pos, best_scores, best_iters, parameters
    
    def _readHparamsLims(
        self,
    ):
        """
        Read hparams lims from data_dir.
        Returns:
            hparams_lims: dictionary of hparams limits { hparam_name: [min, max] }; dictionary { str: [float, float] }
        """
        # read json file
        hparams_lims_file = os.path.join(self.data_dir, 'hparams_lims.json')

        hparams_lims_temp = {}
        with open(hparams_lims_file) as f:
            hparams_lims_temp = json.load(f)

        # flatten dictionary
        hparams_lims = {}
        for elements in hparams_lims_temp.values():
            for hparam, lim in elements.items():
                hparams_lims[hparam] = lim

        return hparams_lims
    




def main():
    data_dir = "results/pso/opt8"
    plotter = PlotterResults(
        data_dir=data_dir,
    )
    plotter.plot()


if __name__ == "__main__":
    main()