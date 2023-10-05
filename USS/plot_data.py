import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



def convertColName(col_name):
    dist = float(col_name.split("_")[0][:-1])
    angle = float(col_name.split("_")[1][:-3])
    return dist, angle

def linInterpolate(data, num_fills=20):
    d = [np.linspace(data[i], data[i+1], num_fills) for i in range(len(data)-1)]
    return np.array(d).flatten()

def main():
    sensor = "MB1603"
    detection_thr = 1.0

    df = pd.read_csv(os.path.join("data", sensor+".csv"))

    # get distances and angles
    dists = []
    angles = []
    for col in df.columns:
        dist, angle = convertColName(col)
        dists.append(dist)
        angles.append(angle)
    dists = np.sort(np.unique(dists))
    angles = np.sort(np.unique(angles))

    # # create column names array
    # cols = np.full((len(dists), len(angles)), "-", dtype=str)
    # for col in df.columns:
    #     print(col)
    #     dist, angle = convertColName(col)
    #     i = np.where(dists == dist)[0][0]
    #     j = np.where(angles == angle)[0][0]
    #     print(f"i: {i}, j: {j}")
    #     cols[i,j] = col

    # get mean, std and ratio for each distance and angle
    means = np.zeros((len(dists), len(angles)), dtype=float)
    stds = np.zeros((len(dists), len(angles)), dtype=float)
    ratios = np.zeros((len(dists), len(angles)), dtype=float)
    for i, dist in enumerate(dists):
        for j, angle in enumerate(angles):
            meas = df[f"{dist}m_{int(angle)}deg"].values
            # meas_val = meas[(meas > dist*(1-detection_thr)) & (meas < dist*(1+detection_thr))]
            meas_val = meas

            means[i,j] = np.mean(meas_val)
            stds[i,j] = np.std(meas_val)
            ratios[i,j] = len(meas_val)/len(meas)

            pass

    # Create colormap
    # cmap = plt.cm.get_cmap('plasma')
    cmap = plt.colormaps.get_cmap('viridis')
    cNorm  = plt.Normalize(vmin=0, vmax=1)

    # Create a figure and axis with polar projection
    fig, axis = plt.subplots(ncols=len(dists), subplot_kw={'projection': 'polar'}, figsize=(15,5))

    r_max = np.max(means + stds)
    a = np.deg2rad(angles)
    # a = np.deg2rad(linInterpolate(data=angles))
    for i, dist in enumerate(dists):
        # m = linInterpolate(data=means[i])
        # s = linInterpolate(data=stds[i])
        # r = linInterpolate(data=ratios[i])
        m = means[i]
        s = stds[i]
        r = ratios[i]

        ax = axis[i]
        colours = cmap(cNorm(r))
        for j in range(len(a)-1):
            print(f"i: {i}, j: {j}")
            print(f"a: {a[i:i+2]}, m: {m[i:i+2]}, s: {s[i:i+2]}")
            ax.plot(a[j:j+2], m[j:j+2], '-', color=colours[j])

            vertices = [(a[j],m[j]-s[j]), 
                        (a[j],m[j]+s[j]), 
                        (a[j+1],m[j+1]+s[j+1]), 
                        (a[j+1],m[j+1]-s[j+1])]
            ax.add_patch(
                Polygon(vertices, closed=False, facecolor=colours[i], edgecolor=None, alpha=0.3)
            )


        ax.set_theta_offset(np.pi / 2)  # Set the zero angle at the top
        ax.set_thetamin(-90)
        ax.set_thetamax(90)

        ax.set_xticks(np.deg2rad([-90, -60, -30, 0, 30, 60, 90]))
        ax.set_xticklabels(['-90°', '-60°', '-30°', '0°', '30°', '60°', '90°'])
        ax.set_thetagrids(angles=np.linspace(-90, 90, 19), weight='black', alpha=0.5)
        # ax.set_rgrids(radii=np.linspace(0, 0.8, 9), weight='black', alpha=0.5)
        ax.set_ylim([0,r_max])

    # ax.legend(loc='upper right')
    plt.show()




if __name__ == '__main__':
    main()