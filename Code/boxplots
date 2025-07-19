import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os


#errors

shot_1 = [20, 131, 135, 36, 97, 60, 75]

shot_2 = [120, 93, 112, 84, 128, 47, 63, 79, 27, 6, 32, 38]

shot_3 = [61, 15, 67, 27, 53, 59, 59, 57, 19, 57, 50, 99]

shot_4 = [45, 3, 45, 27, 8, 77, 37, 57, 17, 15, 9, 19]

throw_1 = [105, 106, 111, 116, 96, 89, 101, 108, 61]

throw_2 = [62, 72, 60, 114, 24, 31, 107 , 91, 16]

throw_3 = [43, 30, 68, 91, 32, 52, 48, 74, 85]

throw_4 = [64, 18, 7, 37, 24, 38, 39, 5, 61]

#real positions

throws_real_x = [30, 90, 150, 30, 90, 150, 30, 90, 150]
throws_real_y =  [110, 110, 110, 80, 80, 80, 40, 40, 40]

shots_real_1_x = [80, 76, 65, 100, 119, 108, 134]
shots_real_1_y = [50, 38, 51, 34, 50, 68, 72]

shots_real_2_3_x = [27, 79, 81, 132, 86, 56, 104, 97, 89, 114, 78, 123]
shots_real_2_3_y = [64, 45, 19, 45, 87, 16, 44, 70, 77, 42, 42, 29]

shots_real_4_x = [63, 114, 59, 88, 151, 175, 54, 60, 60, 118, 114, 39]
shots_real_4_y = [44, 66, 36, 70, 60, 55, 52, 50, 67, 89, 65, 71]

#estimated positios


est_shots_1_x = [95, 173, 190, 112, 190, 161, 112]
est_shots_1_y = [63, 125, 0, 0, 116, 95, 0]

est_shots_4_x = [73, 112, 76, 114, 152, 100, 84, 103, 75, 113, 123, 57]
est_shots_4_y = [89, 65, 78, 79, 52, 37, 73, 88, 60, 103, 64, 74]


#---------------------------------Throw boxplot-------------------------------------------------
plt.figure(figsize=(6, 6))
plt.boxplot([throw_1,  throw_2, throw_3, throw_4], 
           labels=["Stage 1",  "Stage 2", "Stage 3", "Stage 4"])

plt.ylabel("Error Distance / cm", fontsize=20)
plt.grid(True, axis='y')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(0, 140)
plt.tight_layout()
plt.show()
 

#----------------------------------------Shots boxplot------------------------------------
plt.figure(figsize=(6, 6))
plt.boxplot([shot_1, shot_2, shot_3, shot_4],
            labels=["Stage 1", "Stage 2", "Stage 3", "Stage 4"])

plt.ylabel("Error Distance / cm", fontsize=20)
plt.grid(True, axis='y')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(0, 140)
plt.tight_layout()
plt.show()



#---------------------------------scatter----------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 7))

#stage 1
ax.scatter(shots_real_1_x, shots_real_1_y, c='blue', label='Stage 1 - Real',
           marker='o', s=70, edgecolors='black', linewidths=0.5)
ax.scatter(est_shots_1_x, est_shots_1_y, c='blue', label='Stage 1 - Estimated',
           marker='x', s=70, linewidths=1.2)
for x0, y0, x1, y1 in zip(shots_real_1_x, shots_real_1_y, est_shots_1_x, est_shots_1_y):
    ax.plot([x0, x1], [y0, y1], linestyle='--', color='blue', alpha=0.5)

#stage 4
ax.scatter(shots_real_4_x, shots_real_4_y, c='green', label='Stage 4 - Real',
           marker='o', s=70, edgecolors='black', linewidths=0.5)
ax.scatter(est_shots_4_x, est_shots_4_y, c='green', label='Stage 4 - Estimated',
           marker='x', s=70, linewidths=1.2)
for x0, y0, x1, y1 in zip(shots_real_4_x, shots_real_4_y, est_shots_4_x, est_shots_4_y):
    ax.plot([x0, x1], [y0, y1], linestyle='-', color='green', alpha=0.5)

wall_top_left = (0, 126.5)
wall_top_right = (190, 126.5)
wall_bottom_left = (0, 0)
wall_bottom_right = (190, 0)

ax.plot([wall_top_left[0], wall_top_right[0]], [wall_top_left[1], wall_top_right[1]], 'k', linewidth=2)
ax.plot([wall_bottom_left[0], wall_bottom_right[0]], [wall_bottom_left[1], wall_bottom_right[1]], 'k', linewidth=2)
ax.plot([wall_top_left[0], wall_bottom_left[0]], [wall_top_left[1], wall_bottom_left[1]], 'k', linewidth=2)
ax.plot([wall_top_right[0], wall_bottom_right[0]], [wall_top_right[1], wall_bottom_right[1]], 'k', linewidth=2)

#legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Real Position',
           markerfacecolor='gray', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='x', color='gray', label='Estimated Position',
           markersize=8, linewidth=1.5),
    Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Stage 1'),
    Line2D([0], [0], color='green', lw=2, linestyle='-', label='Stage 4'),
]

legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.05, 0.95),
                   fontsize=14, frameon=True)
legend.get_frame().set_alpha(0.9)

legend.get_frame().set_alpha(0.9)

ax.set_xlabel("X / cm", fontsize=20)
ax.set_ylabel("Y / cm", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()


plt.show()
