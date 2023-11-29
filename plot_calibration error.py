import numpy as np
import matplotlib.pyplot as plt


error = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
mnn = [0.115, 0.148, 0.119, 0.122, 0.115, 0.124, 0.119, 0.119]
convergence_50 = [29.8, 21.1, 23.8, 23.8, 23.8, 23.7, 23.8, 74.7]
convergence_25 = [83.8, 23.8, 39.8, 29.8, 26.7, 29.7, 32.9, 74.7]
convergence_10 = [111.4, 23.8, 91.7, 110.2, 96.7, 115.8, 105.4, 117.4]

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