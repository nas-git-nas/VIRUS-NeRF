import torch
import os
import sys
import pandas as pd
from icecream import ic
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import matplotlib.ticker as mtick
 
sys.path.insert(0, os.getcwd())
from training.trainer_plot import TrainerPlot


colors = {
    'robot':    'red',
    'GT_map':   'grey', 
    'GT_scan':  'black',
    'NeRF':     'darkorange',
    'LiDAR':    'darkmagenta',
    'USS':      'blue',
    'ToF':      'green',
}

zone_lims = {
    "zone1": [0, 1],
    "zone2": [0, 2],
    "zone3": [0, 100],
}

def loadAblationStudy(
    base_dir,
    seeds,
):
    """
    Load ablation study results
    Args:
        base_dir: base directory; str
        seeds: seeds; list
    Returns:
        sensors_dict_list: list of dictionaries with results; list
    """
    metrics = [
        'nn_mean', 'nn_mean_inv', 'nn_mean_inv_360', 
        'nn_median', 'nn_median_inv', 'nn_median_inv_360', 
        'nn_inlier', 'nn_inlier_inv', 'nn_inlier_inv_360',
        'nn_outlier_too_close', 'nn_outlier_too_close_inv', 'nn_outlier_too_close_inv_360',
    ]

    sensors_dict_list = []
    for seed in seeds:
        metric_file = os.path.join(base_dir, f"seed_{seed}", "metrics.csv")
        df = pd.read_csv(metric_file, index_col=[0])

        sensors_dict = {}
        for sensor in ['NeRF', 'LiDAR', 'USS', 'ToF']:
            sensors_dict[sensor] = {}
            for metric in metrics:
                zone_str =df.loc[sensor, metric]
                zone_str = zone_str.replace("'", '"')
                zone_dict = json.loads(zone_str)
                sensors_dict[sensor][metric] = zone_dict

        sensors_dict_list.append(sensors_dict)
        
    # ic(sensors_dict_list)
    return sensors_dict_list

def plotMultipleMetrics(
    metrics_dict_list:list,
    colors:dict,
    zone_lims:list,
    base_dir:str,
):
    """
    Plot average metrics.
    Args:
        metrics_dict: list with multiple dict of metrics
    """   
    sensors = list(metrics_dict_list[0].keys())
    zones = list(metrics_dict_list[0][sensors[0]]['nn_mean'].keys())

    x = np.arange(len(zones))  # the label locations
    width = 0.6  # the width of the bars

    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(13,8), gridspec_kw={'width_ratios': [5.5, 5.5, 3.5]})
    metrics = [
        'nn_mean', 'nn_mean_inv', 'nn_mean_inv_360', 
        'nn_median', 'nn_median_inv', 'nn_median_inv_360', 
        'nn_inlier', 'nn_inlier_inv', 'nn_inlier_inv_360',
    ]

    y_axis_inv_mean_max = 0.0
    y_axis_inv_median_max = 0.0
    for i, (ax, metric) in enumerate(zip(axs.flatten(), metrics)):

        for j, sensor in enumerate(sensors):

            x_axis = x - width/2 + (j+0.5)*width/len(sensors)

            multiple_performances = np.zeros((len(metrics_dict_list), len(zones)))
            multiple_nn_outlier_too_close = np.zeros((len(metrics_dict_list), len(zones)))
            for k, metrics_dict in enumerate(metrics_dict_list):
                multiple_performances[k] = np.array([metrics_dict[sensors[j]][metric][z] for z in zones])
                if '360' in metric:
                    too_close_metric_name = 'nn_outlier_too_close_inv_360'
                elif 'inv' in metric:
                    too_close_metric_name = 'nn_outlier_too_close_inv'
                else:
                    too_close_metric_name = 'nn_outlier_too_close'
                multiple_nn_outlier_too_close[k] = np.array([metrics_dict[sensors[j]][too_close_metric_name][z] for z in zones])

            performances_mean = np.mean(multiple_performances, axis=0)
            performances_std = np.std(multiple_performances, axis=0)
            nn_outlier_too_close_mean = np.mean(multiple_nn_outlier_too_close, axis=0)
            nn_outlier_too_far_mean = 1 - performances_mean - nn_outlier_too_close_mean

            if i < 6:
                if (i%3) != 0:
                    if i < 3:
                        y_axis_inv_mean_max = max(y_axis_inv_mean_max, np.max(performances_mean+performances_std))
                    else:
                        y_axis_inv_median_max = max(y_axis_inv_median_max, np.max(performances_mean+performances_std))

                if (i+1) % 3 == 0:
                    ax.bar(x_axis, performances_mean, width/len(sensors), color=colors[sensor])
                else:
                    sensor_label = sensor
                    if sensor == "ToF":
                        sensor_label = "IRS"
                    ax.bar(x_axis, performances_mean, width/len(sensors), label=sensor_label, color=colors[sensor])
            else:
                if (((i + j) % 2) == 0) and (i < 8):
                    ax.bar(x_axis, performances_mean, width/len(sensors), label='Inliers', color=colors[sensor])
                    ax.bar(x_axis, nn_outlier_too_close_mean, width/len(sensors), bottom=performances_mean, 
                            label='Outliers \n(too close)', color=colors[sensor], alpha=0.4)
                    ax.bar(x_axis, nn_outlier_too_far_mean, width/len(sensors), bottom=1-nn_outlier_too_far_mean, 
                            label='Outliers \n(too far)', color=colors[sensor], alpha=0.1)
                else:
                    ax.bar(x_axis, performances_mean, width/len(sensors), color=colors[sensor])
                    ax.bar(x_axis, nn_outlier_too_close_mean, width/len(sensors), bottom=performances_mean, 
                            color=colors[sensor], alpha=0.4)
                    ax.bar(x_axis, nn_outlier_too_far_mean, width/len(sensors), bottom=1-nn_outlier_too_far_mean, 
                            color=colors[sensor], alpha=0.1)
                    
            ax.errorbar(x_axis, performances_mean, yerr=performances_std, fmt='none', ecolor="black", capsize=2)

        if (i+1) % 3 == 0:  
            ax.set_xlim([-0.75*width, np.max(x)+0.75*width])
        else: 
            ax.set_xlim([-0.75*width, np.max(x)+2.75*width])
            ax.legend()

        if i < 6:
            ax.set_xticks(x, [])
        else:
            ax.set_xticks(x, [f"{zone_lims[z][0]}-{zone_lims[z][1]}m" for z in zones])
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))


    axs[0,1].set_ylim([0.0, 1.05*y_axis_inv_mean_max])
    axs[0,2].set_ylim([0.0, 1.05*y_axis_inv_mean_max])
    axs[1,1].set_ylim([0.0, 1.05*y_axis_inv_median_max])
    axs[1,2].set_ylim([0.0, 1.05*y_axis_inv_median_max])
    axs[2,0].set_ylim([0.0, 1.05])
    axs[2,1].set_ylim([0.0, 1.05])
    axs[2,2].set_ylim([0.0, 1.05])
    axs[0,0].set_ylabel('Mean [m]')
    axs[1,0].set_ylabel('Median [m]')
    axs[2,0].set_ylabel('Inliers [%]')
    axs[0,0].set_title('Accuracy: Sensor->GT(FoV)') 
    axs[0,1].set_title('Coverage: GT(FoV)->Sensor') 
    axs[0,2].set_title('Coverage: GT(360°)->Sensor') 

    fig.suptitle('Nearest Neighbour Distance', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"metrics.png"))

def plot_ablation_study():
    base_dir = "results/ETHZ/ablation/optimized"
    num_trainings = 10
    base_seed = 21
    seeds = np.arange(base_seed, base_seed+num_trainings)

    sensors_dict_list = loadAblationStudy(
        base_dir=base_dir,
        seeds=seeds,
    )

    plotMultipleMetrics(
        metrics_dict_list=sensors_dict_list,
        colors=colors,
        zone_lims=zone_lims,
        base_dir=base_dir,
    )


if __name__ == "__main__":
    plot_ablation_study()