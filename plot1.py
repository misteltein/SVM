import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("sample1.csv",delimiter=",",dtype={'names':('x','y','c'),'formats':(np.float64,np.float,np.int8)})

x=data['x']
y=data['y']
c=data['c']

plt.figure()
plt.scatter(x,y,c=c)
plt.show()
