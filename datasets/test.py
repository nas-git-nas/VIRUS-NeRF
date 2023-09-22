import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_points_and_directions(x, y, z, yaw, pitch, roll):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(x, y, z, label='Points')
    
    # Plot directions
    for i in range(len(x)):
        x_dir = np.cos(yaw[i]) * np.cos(pitch[i])
        y_dir = np.sin(yaw[i]) * np.cos(pitch[i])
        z_dir = np.sin(pitch[i])
        ax.quiver(x[i], y[i], z[i], x_dir, y_dir, z_dir, length=0.1, normalize=True, color='r', label='Direction')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Example usage
N = 10  # Number of points
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)
yaw = np.random.rand(N) * 2 * np.pi  # Random yaw angles in radians
pitch = np.random.rand(N) * np.pi - np.pi/2  # Random pitch angles in radians
roll = np.random.rand(N) * 2 * np.pi  # Random roll angles in radians

plot_points_and_directions(x, y, z, yaw, pitch, roll)
