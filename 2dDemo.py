import numpy as np
import sklearn.svm as svm
import matplotlib.pyplot as plt

X = np.random.randn(100, 2)
y = X[:,0] + 2*X[:,1] > 1

model = svm.LinearSVC()
model.fit(X,y)

slope = -model.coef_[0,1] / model.coef_[0,0]

#line: ax + by + c = 0
a = model.coef_[0,0]
b = model.coef_[0,1]
c = model.intercept_
line_x = np.array([-2.0,+2.0])
line_y = -(a/b)*line_x - (c/b)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(X[:, 0], X[:, 1],c=y)
ax.plot(line_x,line_y)
plt.show()
