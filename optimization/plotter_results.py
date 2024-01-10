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
        pos, scores, parameters = self._readPosData(self.data_dir)
        best_pos, best_scores, best_parameters = self._readBestPosData(self.data_dir)
        hparams_lims = self._readHparamsLims()

        # verify that parameters are the same
        assert parameters == best_parameters

        # keep only best N scores
        idxs = np.argsort(best_scores)
        best_pos = best_pos[idxs[:self.keep_best_n], :]
        best_scores = best_scores[idxs[:self.keep_best_n]]

        # normalize pos and scores
        for i, param in parameters.items():
            pos[:, i] = (pos[:, i] - hparams_lims[param][0]) / (hparams_lims[param][1] - hparams_lims[param][0])
            best_pos[:, i] = (best_pos[:, i] - hparams_lims[param][0]) / (hparams_lims[param][1] - hparams_lims[param][0])

        best_scores_norm = np.where(best_scores > self.score_max, self.score_max, best_scores)
        best_scores_norm = np.where(best_scores_norm < self.score_min, self.score_min, best_scores_norm)
        scores_norm = np.where(scores > self.score_max, self.score_max, scores)
        scores_norm = np.where(scores_norm < self.score_min, self.score_min, scores_norm)
        best_scores_norm = (best_scores_norm - np.min(scores_norm)) / (np.max(scores_norm) - np.min(scores_norm))
        scores_norm = (scores_norm - np.min(scores_norm)) / (np.max(scores_norm) - np.min(scores_norm))

        # plot
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(12, 8))
        column_width = 0.35

        # reverse colotmap
        cmap = matplotlib.colormaps['jet']
        cmap_inv = cmap.reversed() 

        ax = axes
        for i, param in parameters.items():
            im = ax.scatter(i + column_width*(scores_norm-0.5), pos[:, i], c=scores, 
                            cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max)
            
            # plot star for best score
            for j in range(self.keep_best_n):
                ax.scatter(i + column_width*(best_scores_norm[j]-0.5), best_pos[j, i], c=best_scores[j],
                        cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[j], s=200)
            # ax.scatter(i + column_width*(best_scores_norm-0.5), best_pos[:, i], c=best_scores,
            #            cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs, s=200)

        ax.set_xticks(list(parameters.keys()))
        ax.set_xticklabels([param.replace('_', ' ') + f":\n     [{hparams_lims[param][0]:.2f}, {hparams_lims[param][1]:.2f}]" 
                            for param in parameters.values()])
        ax.set_ylabel('Normalized Hyper-Parameters')
        ax.set_title('Particle Swarm Optimization')
        fig.autofmt_xdate()

        # add colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8]) # [left, bottom, width, height]
        # cbar_ax.set_title('Mean NND',fontsize=13)
        fig.colorbar(im, cax=cbar_ax)

        plt.show()

    def _readPosData(
            self,
            data_dir,
        ):
        """
        Read position data from data_dir.
        Returns:
            pos: particle positions; numpy array of shape (N, M)
            scores: particle scores; numpy array of shape (N,)
            parameters: dictionary of parameters { column_index: parameter_name}; dictionary { int: str }
        """
        # get position files
        files = []
        for file in os.listdir(self.data_dir):
            if 'pso_pos' in file:
                files.append(file)

        # read parameters and number of iterations
        df = pd.read_csv(os.path.join(data_dir, files[0]))
        columns = df.columns.to_list()
        columns.remove('score')
        columns.remove('time')
        columns.remove('iteration')
        parameters = { i:param for i, param in enumerate(columns) }

        # read data
        pos = np.zeros((0, len(columns)))
        scores = np.zeros((0, 1))

        for file in files:
            df = pd.read_csv(os.path.join(data_dir, file))
            pos_temp = df[columns].to_numpy()
            scores_temp = df[['score']].to_numpy()

            pos = np.concatenate((pos, pos_temp), axis=0)
            scores = np.concatenate((scores, scores_temp), axis=0)

        scores = scores.flatten()
        return pos, scores, parameters
    
    def _readBestPosData(
            self,
            data_dir,
        ):
        """
        Read position data from data_dir.
        Returns:
            pos: particle positions; numpy array of shape (N, M)
            scores: particle scores; numpy array of shape (N,)
            parameters: dictionary of parameters { column_index: parameter_name}; dictionary { int: str }
        """
        # get position files
        files = []
        for file in os.listdir(self.data_dir):
            if 'pso_best_pos' in file:
                files.append(file)

        # read parameters and number of iterations
        df = pd.read_csv(os.path.join(data_dir, files[0]))
        columns = df.columns.to_list()
        columns.remove('best_score')
        columns.remove('best_count')
        parameters = { i:param for i, param in enumerate(columns) }

        # read data
        best_pos = np.zeros((0, len(columns)))
        best_scores = np.zeros((0, 1))
        for file in files:
            df = pd.read_csv(os.path.join(data_dir, file))
            pos_temp = df[columns].to_numpy()
            scores_temp = df[['best_score']].to_numpy()

            best_pos = np.concatenate((best_pos, pos_temp), axis=0)
            best_scores = np.concatenate((best_scores, scores_temp), axis=0)

        # keep only last occurance of each best position
        best_pos_inv = best_pos[::-1, :]
        best_scores_inv = best_scores.flatten()[::-1]
        best_pos_inv, idxs = np.unique(best_pos_inv, axis=0, return_index=True)
        best_scores_inv = best_scores_inv[idxs]
        best_pos = best_pos_inv[::-1, :]
        best_scores = best_scores_inv[::-1]

        return best_pos, best_scores, parameters
    
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