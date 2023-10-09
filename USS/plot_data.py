import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



def convertColName(col_name):
    dist = float(col_name.split("_")[0][:-1])
    if dist == 1.0 or dist == 2.0:
        dist = int(dist)
    angle = float(col_name.split("_")[1][:-3])
    return dist, angle

def linInterpolate(data, num_fills=20):
    d = [np.linspace(data[i], data[i+1], num_fills) for i in range(len(data)-1)]
    return np.array(d).flatten()

def correctMeas(meas):
    return meas - 0.04

def loadData(sensor, object, surface):
    file_name = sensor + '_' + object
    if surface == 'plexiglas':
        file_name += '_plex'        
    return pd.read_csv(os.path.join("data", file_name+".csv"))

def main():
    sensor = "MB1603" # either "URM37", "HC-SR04" or "MB1603"

    # the relative mean absolute error of valid measurements must be within 50% of the target distance
    detection_thr = 0.5

    # get distances and angles
    dists = [0.25, 0.5, 1.0, 2.0]
    angles = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    objects = ['large', 'medium', 'small']
    surfaces = ['cardboard', 'plexiglas']

    # Create colormap
    cmap = plt.colormaps.get_cmap('plasma')
    cNorm  = plt.Normalize(vmin=0, vmax=0.9)

    # Create a figure and axis with polar projection
    fig, axis = plt.subplots(ncols=len(surfaces), nrows=len(objects), subplot_kw={'projection': 'polar'}, figsize=(10,9))

    for k, object in enumerate(objects):
        for l, surface in enumerate(surfaces):
            ax = axis[k,l]

            # load dataframe
            df = loadData(sensor=sensor, object=object, surface=surface)

            # get mean, std and ratio for each distance and angle
            means = np.zeros((len(dists), len(angles)), dtype=float)
            stds = np.zeros((len(dists), len(angles)), dtype=float)
            ratios = np.zeros((len(dists), len(angles)), dtype=float)
            ma_error = np.zeros((len(dists), len(angles)), dtype=float)
            rma_error = np.zeros((len(dists), len(angles)), dtype=float)
            for i, dist in enumerate(dists):
                for j, angle in enumerate(angles):
                    if f"{dist}m_{int(angle)}deg" in df.columns:
                        meas = df[f"{dist}m_{int(angle)}deg"].values
                    elif f"{int(dist)}m_{int(angle)}deg" in df.columns:
                        meas = df[f"{int(dist)}m_{int(angle)}deg"].values
                    else:
                        ratios[i,j] = 0
                        continue

                    meas = correctMeas(meas=meas)
                    meas_val = meas
                    # meas_val = meas[(meas > dist*(1-detection_thr)) & (meas < dist*(1+detection_thr))]
                    # if len(meas_val) == 0:
                    #     ratios[i,j] = 0
                    #     continue

                    means[i,j] = np.mean(meas_val)
                    stds[i,j] = np.std(meas_val)
                    ratios[i,j] = len(meas[(meas > dist*(1-detection_thr)) & (meas < dist*(1+detection_thr))])/len(meas)
                    ma_error[i,j] = np.mean(np.abs(meas_val - dist))
                    rma_error[i,j] = np.mean(np.abs(meas_val - dist)) / dist

            

            # a = np.deg2rad(angles)
            a = np.deg2rad(linInterpolate(data=angles))
            for i, dist in enumerate(dists):
                m = linInterpolate(data=means[i])
                s = linInterpolate(data=stds[i])
                r = linInterpolate(data=rma_error[i])

                # m = means[i]
                # s = stds[i]
                # r = ratios[i]                
                
                colours = cmap(cNorm(ma_error[i]))
                colours = np.concatenate((linInterpolate(data=colours[:,0]).reshape(-1,1), 
                                          linInterpolate(data=colours[:,1]).reshape(-1,1), 
                                          linInterpolate(data=colours[:,2]).reshape(-1,1),
                                          linInterpolate(data=colours[:,3]).reshape(-1,1)), axis=1)
                for j in range(len(a)-1):

                    # if ratios[i,j] < valid_thr or ratios[i,j+1] < valid_thr:
                    #     continue
                    # if rma_error[i,j] > 0.5 or rma_error[i,j+1] > 0.5:
                    #     continue
                    if r[j] > detection_thr or r[j+1] > detection_thr:
                        continue

                    ax.plot(a[j:j+2], m[j:j+2], '-', color=colours[j])

                    vertices = [(a[j],m[j]-s[j]), 
                                (a[j],m[j]+s[j]), 
                                (a[j+1],m[j+1]+s[j+1]), 
                                (a[j+1],m[j+1]-s[j+1])]
                    ax.add_patch(
                        Polygon(vertices, closed=False, facecolor=colours[j], edgecolor=None, alpha=0.5)
                    )


            ax.set_theta_offset(np.pi / 2)  # Set the zero angle at the top
            ax.set_thetamin(-40)
            ax.set_thetamax(40)

            ax.set_xticks(np.deg2rad([-40, -20, 0, 20, 40]), labels=None) 
            ax.set_yticks([1.0, 2.0, 3.0], labels=None)
            ax.set_yticklabels(['1m', '2m', '3m'])
            if k == 0:
                ax.set_xticklabels(['-40°', '-20°', '0°', '20°', '40°'])
            else:
                ax.set_xticklabels([])
            # if k == len(objects)-1:
            #     ax.set_xlabel('Distance [m]', fontsize=12, y=0.7)

            ax.set_thetagrids(angles=[-40, -30, -20, -10, 0, 10, 20, 30, 40], weight='black', alpha=0.5, labels=None)
            ax.set_rgrids(radii=[0.25, 0.5, 1.0, 2.0, 3.0], weight='black', alpha=0.5, labels=None)
            ax.set_ylim([0,3])



            if k == 0:
                ax.set_title(surface.capitalize(), weight='bold', y=1.05, fontsize=12)
            if l == 0:
                ax.set_ylabel(object.capitalize(), weight='bold', fontsize=12)



    # ax.legend(loc='upper right')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cNorm)
    sm.set_array(angles)
    cbar = plt.colorbar(sm, ax=axis.ravel().tolist()) 
    cbar.set_label('Mean Absolute Error [m]')  # Label for the colorbar
    # plt.tight_layout()
    plt.subplots_adjust(hspace=-0.15, wspace=-0.2, right=0.75, left=0)
    plt.show()




if __name__ == '__main__':
    main()