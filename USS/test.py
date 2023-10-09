import numpy as np
import matplotlib.pyplot as plt

# Sample data
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
values = [4, 3, 5, 2, 1]

# Create a polar plot
fig, ax = plt.subplots(subplot_kw=dict(polar=True))

# Plot the data
theta = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
ax.plot(theta, values, label='Data')

# Set the radial gridlines
ax.set_rgrids([0, 2, 4, 6, 8], angle=45)

# Manually position and rotate the radial tick labels
ax.set_yticks([1, 2, 3, 4, 5])  # Set the positions
ax.set_yticklabels(categories)  # Set the labels

# Rotate the labels
for label in ax.yaxis.get_ticklabels():
    label.set_rotation(45)

# Add legend and title
ax.legend()
ax.set_title('Polar Plot with Rotated Radial Labels')

plt.show()



