import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from info import minmax
from info import plane

data = np.loadtxt("sample1.csv",delimiter=",")

P=data[:,0:3]
x=P[:,0]
y=P[:,1]
z=P[:,2]
c=data[:,3]

model = SVC(kernel='linear',random_state=None)
model.fit(P,c)
w = model.coef_[0]
b = model.intercept_

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(x,y,z,c=c)
ax.plot_surface(*plane(w,b,minmax(x),minmax(y)),alpha=0.5)
plt.show()

