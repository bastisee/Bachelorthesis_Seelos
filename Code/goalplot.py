import numpy as np
import matplotlib.pyplot as plt

# Wall dimensions
wall_width = 190
wall_height = 126.5
wall_z = 0

wall_top_left = (0, 126.5)
wall_top_right = (190, 126.5)
wall_bottom_left = (0, 0)
wall_bottom_right = (190, 0)

# Microphone positions [x, y, z] (cm)
mic1 = np.array([22.5, 118, -2.5])   # Mic 1 (left)
mic2 = np.array([167.5, 118, -2.5])   # Mic 2 (right)
mic3 = np.array([95, 17.5, -2.5])   #Mic 3 bottom




#------Plot------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

# Plot the wall
ax.plot([wall_top_left[0], wall_top_right[0]], [wall_top_left[1], wall_top_right[1]], 'k', linewidth=3, label="Wall Outline")
ax.plot([wall_bottom_left[0], wall_bottom_right[0]], [wall_bottom_left[1], wall_bottom_right[1]], 'k', linewidth=3)
ax.plot([wall_top_left[0], wall_bottom_left[0]], [wall_top_left[1], wall_bottom_left[1]], 'k', linewidth=3)
ax.plot([wall_top_right[0], wall_bottom_right[0]], [wall_top_right[1], wall_bottom_right[1]], 'k', linewidth=3)

#Plot Mics
# Plot Mics
ax.text(mic1[0], mic1[1], r"$x_1$", fontsize=14, ha='center', va='center')
ax.text(mic2[0], mic2[1], r"$x_2$", fontsize=14, ha='center', va='center')
ax.text(mic3[0], mic3[1], r"$x_3$", fontsize=14, ha='center', va='center')

# legend
mic1_legend = plt.Line2D([0], [0], linestyle="none", marker='', label=f"$x_1$ ({mic1[0]}, {mic1[1]}, {mic1[2]})")
mic2_legend = plt.Line2D([0], [0], linestyle="none", marker='', label=f"$x_2$ ({mic2[0]}, {mic2[1]}, {mic2[2]})")
mic3_legend = plt.Line2D([0], [0], linestyle="none", marker='', label=f"$x_3$ ({mic3[0]}, {mic3[1]}, {mic3[2]})")
wall_legend = plt.Line2D([0], [0], color='black', linewidth=3, label="Wall Outline")

# Labels and formatting

ax.set_xlabel("X / cm")
ax.set_ylabel("Y / cm")
ax.set_aspect('equal')
ax.grid(True)
ax.legend(handles=[wall_legend, mic1_legend, mic2_legend, mic3_legend], loc="best")


# Show plot
plt.show()