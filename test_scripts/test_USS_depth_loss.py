import numpy as np
import matplotlib.pyplot as plt

def calcLosses(x, meas):
    depths_w = np.exp( -(x - np.min(x))/0.1 )
    # depths_w = depths_w / np.sum(depths_w)
    losses = np.abs(x - meas)
    depth_losses = depths_w * losses

    return losses, depths_w, depth_losses

def plotLosses(ax, x, losses, depths_w, depth_losses):
    ax.plot(x, losses, label="losses", color="blue")
    ax.scatter(x, depths_w, label="w", color="green")
    ax.scatter(x, depth_losses, label="result", color="red")
    ax.legend()

def main():
    num_points = 100
    fig, axis = plt.subplots(1, 1)
    meas = 0

    x = np.linspace(0, 1, num_points) + np.random.rand(num_points)/10
    losses, depths_w, depth_losses = calcLosses(x, meas)
    print(f"1, sum of depth losses: {np.sum(depth_losses)}")

    ax = axis
    plotLosses(ax=axis, x=x, losses=losses, depths_w=depths_w, depth_losses=depth_losses)
    ax.set_ylim([0, 1])


    plt.show()


if __name__ == "__main__":
    main()