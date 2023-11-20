import numpy as np
import matplotlib.pyplot as plt


error = [0.0, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0]
mnn = [0.111 , 0.103, 0.094, 0.101, 0.104, 0.112, 0.095]
convergence_50 = [36.4, 34.9, 31.4, 45.7,27.9, 24.8, 42.6]
convergence_25 = [83.7, 72.8, 42.4, 63.1, 80.7, 92.0, 72.9]
convergence_10 = [83.7, 87.5, 87.6, 76.8, 121.8, 99.3, 115.8]

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
x_axis = np.arange(len(error))
 
ax = axes[0]
ax.bar(x_axis, mnn, color ='blue', width = 0.4)
ax.set_xlabel("Added angle error (degree)")
ax.set_ylabel("Mean Nearest-Neighbor Distance [m]")
ax.set_xticks(x_axis, error)

ax = axes[1]
ax.bar(x_axis - 0.2, convergence_50, color ='blue', width = 0.2, label='50%')
ax.bar(x_axis, convergence_25, color ='orange', width = 0.2, label='25%')
ax.bar(x_axis + 0.2, convergence_10, color ='green', width = 0.2, label='10%')
ax.set_xlabel("Added angle error (degree)")
ax.set_ylabel("Convergence time [s]")
ax.legend()
ax.set_xticks(x_axis, error)
 
plt.show()