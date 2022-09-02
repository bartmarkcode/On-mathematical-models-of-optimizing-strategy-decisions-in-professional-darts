from matplotlib import pyplot as plt
import numpy as np

# ================================================================================
# DRAW a dartboard with polar coordinate labels:

fig = plt.figure()
plt.rc('xtick', labelsize=12)
plt.rc("font", weight="bold")
ax = fig.add_subplot(111, projection="polar")
ax.set_xticks(np.arange(np.pi / 20, 2 * np.pi + np.pi / 20, np.pi / 10))
ax.tick_params(pad=8)
ax.set_yticks(np.array([6.4, 15.9, 99, 107, 162, 170, 190]))
# ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

# ax_image = fig.add_subplot(111)
# ax_image.imshow(board,alpha = 0.6)
# ax_image.axis("off")
# ax.set_thetamin(90)
# ax.set_thetamax(180)

plt.show()
