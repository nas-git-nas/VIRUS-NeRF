import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker

from matplotlib.colors import LinearSegmentedColormap

# Sample data (replace with your actual data)
angles = np.deg2rad([-90, -45, 0, 45, 90])  # Adjusted angles from -90 to 90 degrees
mean_values = np.array([10, 20, 15, 25, 30])  # Mean values for each angle
std_values = np.array([2, 3, 1, 4, 2])  # Standard deviation for each angle
val_ratios = np.array([0.9, 0.5, 0.7, 0.6, 0.1])


num_fills = 20
angles_l = np.empty(0)
means_l = np.empty(0)
std_l = np.empty(0)
ratios_l = np.empty(0)
for i in range(len(angles)-1):
    angles_l = np.concatenate((angles_l, np.linspace(angles[i], angles[i+1], num_fills)))
    means_l = np.concatenate((means_l, np.linspace(mean_values[i], mean_values[i+1], num_fills)))
    std_l = np.concatenate((std_l, np.linspace(std_values[i], std_values[i+1], num_fills)))
    ratios_l = np.concatenate((ratios_l, np.linspace(val_ratios[i], val_ratios[i+1], num_fills)))



# Create colormap
cmap = plt.cm.get_cmap('plasma')
cNorm  = plt.Normalize(vmin=0, vmax=1)



# Create a figure and axis with polar projection
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Plot mean values as dots connected by lines
# ax.plot(angles, mean_values, 'o-', label='Mean Values')
print(f"shape of cmap: {cmap(cNorm(ratios_l)).shape}")
print(f"shape of angles: {angles_l.shape}")

colours = cmap(cNorm(ratios_l))
for i in range(len(angles_l)-1):
    ax.plot(angles_l[i:i+2], means_l[i:i+2], '-', color=colours[i])

    vertices = [(angles_l[i],means_l[i]-std_l[i]), 
                (angles_l[i],means_l[i]+std_l[i]), 
                (angles_l[i+1],means_l[i+1]+std_l[i+1]), 
                (angles_l[i+1],means_l[i+1]-std_l[i+1])]

    # Create a Polygon patch
    polygon = Polygon(vertices, closed=False, facecolor=colours[i], edgecolor=None, alpha=0.3)
    ax.add_patch(polygon)


# Plot error bars for standard deviation
# ax.errorbar(angles, mean_values, yerr=std_values, linestyle='None', color='red', label='Std Deviation')
# ax.fill_between(angles, mean_values - std_values, mean_values + std_values, color='gray', alpha=0.3, label='Std Deviation')

# ax.fill_between(angles_l, means_l - std_l, means_l + std_l, color=cmap(cNorm(ratios_l)), alpha=0.3, label='Std Deviation')



# # Customize the plot

ax.set_theta_offset(np.pi / 2)  # Set the zero angle at the top
ax.set_thetamin(-90)
ax.set_thetamax(90)

ax.set_xticks(np.deg2rad([-90, -60, -30, 0, 30, 60, 90]))
ax.set_xticklabels(['-90°', '-60°', '-30°', '0°', '30°', '60°', '90°'])
ax.set_thetagrids(angles=np.linspace(-90, 90, 19), weight='black', alpha=0.5)


ax.legend(loc='upper right')
plt.title('Mean Values with Standard Deviation as Error Bars')
plt.show()



