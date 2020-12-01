import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("sample1.csv",delimiter=",")

P=data[:,0:3]
c=data[:,3]

from sklearn.svm import SVC
model = SVC(kernel='linear',random_state=None)
model.fit(P,c)

print(model.coef_,model.intercept_)

def minmax(list_):
    min_ = +sys.float_info.max
    max_ = -sys.float_info.max
    for x in list_:
        if min_ > x:
            min_=x
        if max_ < x:
            max_ = x
    return min_, max_

minx, maxx = minmax(P[:,0])
miny, maxy = minmax(P[:,1])
minz, maxz = minmax(P[:,2])

x_plane = np.arange(-3, 3, 0.1)
y_plane = np.arange(-3, 3, 0.1)
xx_plane, yy_plane = np.meshgrid(x_plane, y_plane)
coef=model.coef_[0]
zz_plane = -(coef[0]*xx_plane + coef[1]*yy_plane + model.intercept_) / coef[2]


fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(P[:,0],P[:,1],P[:,2],c=c)
ax.plot_surface(xx_plane, yy_plane, zz_plane)
plt.show()
