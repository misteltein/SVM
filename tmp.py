import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = np.loadtxt('sample.csv',delimiter=',')
data_train, data_test = train_test_split(data,test_size=0.2,random_state=0)
y_train = data_train[:,2]
X_train = data_train[:,:2]
y_test = data_test[:,2]
X_test = data_test[:,:2]

model = svm.SVC(kernel='linear',C=1.0)
model.fit(X_train,y_train)

pred_test = model.predict(X_test)
accuracy_train = accuracy_score(y_test, pred_test)
print('推定の正解率： %.2f' % accuracy_train)

#fig = plt.figure()
#ax = fig.add_subplot()
#ax.set_xlabel('$x_1$')
#ax.set_ylabel('$x_2$')
#plt.scatter(X[:,0],X[:,1],c=y,s=1)
#plt.show()
