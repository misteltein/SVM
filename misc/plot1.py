import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("sample1.csv",delimiter=",")

P=data[:,0:3]
c=data[:,3]

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(P[:,0],P[:,1],P[:,2],c=c)
plt.show()
