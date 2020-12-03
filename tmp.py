import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from info import minmax
from info import plane

data = np.loadtxt("sample2.csv",delimiter=",")

P=data[:,0:3]
x=P[:,0]
y=P[:,1]
z=P[:,2]
c=data[:,3]

model = svm.LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-6)
model.fit(P,c)
w = model.coef_[0]
b = model.intercept_

pred_train = model.predict(P)
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(c, pred_train)
print('トレーニングデータに対する正解率： %.2f' % accuracy_train)

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(x,y,z,c=c)

xx,yy,zz=plane(w,b,minmax(x),minmax(y))

ax.plot_surface(xx,yy,zz,alpha=0.5)

# xx=[1.0 for x in xx]
# ax.plot_surface(xx,yy,zz,alpha=0.5)

plt.show()

