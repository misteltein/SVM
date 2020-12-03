import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("sample1.csv",delimiter=",",dtype={'names':('x','y','c'),'formats':(np.float64,np.float,np.int8)})

x=data['x']
y=data['y']
c=data['c']

P = [[x[i],y[i]]for i in range(len(x))]
print(P)

from sklearn.svm import SVC
model = SVC(kernel='linear',random_state=None)
model.fit(P,c)

print(model.coef_,model.intercept_)

#plt.figure()
#plt.xlabel('x')
#plt.ylabel('y')
#plt.scatter(x,y,c=c)
#plt.show()
