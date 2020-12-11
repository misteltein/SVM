import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = np.loadtxt('sample2.csv',delimiter=',')

cs = np.logspace(-2,8,11,base=10)
gs = np.logspace(-9,3,13,base=10)

scores = {}

for rs in range(0,1,1):
    data_train, data_test = train_test_split(data,test_size=0.25,random_state=rs)
    y_train = data_train[:,2]
    X_train = data_train[:,:2]
    y_test = data_test[:,2]
    X_test = data_test[:,:2]

    for g in gs:
        for c in cs:
            model = svm.SVC(kernel='rbf',C=c,gamma=g)
            model.fit(X_train,y_train)
            pred_test = model.predict(X_test)
            accuracy_train = accuracy_score(y_test, pred_test)
            print(g, c, accuracy_train )
            if (g,c) in scores:
                scores[(g,c)] += accuracy_train
            else:
                scores[(g,c)]=accuracy_train

X1 = []
X2 = []
A = []
for k in scores.keys():
    X1.append(k[0])
    X2.append(k[1])
    A.append(scores[k]/(len(cs)*len(gs)))


import matplotlib.cm as cm
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xscale('log')
ax.set_yscale('log')
sc = ax.scatter(X1,X2,c=A,cmap=cm.coolwarm)
plt.colorbar(sc)
plt.show()
