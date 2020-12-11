import numpy as np
import sklearn.svm as svm
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


data = np.loadtxt('hoo',delimiter=',')

X=data[:,:2]
y=data[:,2].astype(int)

model = svm.SVC()
#model = svm.SVC(kernel='linear',C=1.0)
#model = svm.SVC(gamma='auto')
model.fit(X,y)

fig=plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('x0')
ax.set_ylabel('x1')
plot_decision_regions(X, y, clf=model,  res=0.01)
plt.show()
