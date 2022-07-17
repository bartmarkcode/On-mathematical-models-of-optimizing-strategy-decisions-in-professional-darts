from matplotlib import pyplot as plt
import numpy as np

board = plt.imread("C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/R_Heatmaps/Dartboard_plotted.PNG")

fig = plt.figure()
ax = fig.add_subplot(111,projection ="polar")
ax.set_xticks(np.arange(np.pi/20,2*np.pi+np.pi/20,np.pi/10))
ax.set_yticks(np.array([6.4,15.9,170]))
ax.yaxis.set_ticklabels([])

ax_image = fig.add_subplot(111)
ax_image.imshow(board,alpha = 0.4)
ax_image.axis("off")
plt.show()