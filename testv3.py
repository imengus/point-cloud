import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore")
import sys

mat = np.genfromtxt(sys.argv[1], delimiter=',')
# mat = np.genfromtxt('a.csv', delimiter=',')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mat[:,0], mat[:,1], mat[:,2], alpha=0.07, c='k')

ax.set_xlabel('x')
ax.view_init(elev=0, azim=180)
plt.show()