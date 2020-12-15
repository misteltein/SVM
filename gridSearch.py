import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

sample_name = 'sampleA'
data = np.loadtxt(sample_name+'.csv',delimiter=',')


params = [
        {'C': np.logspace(0,4,9,base=10), 'kernel':['linear']} 
    ]

gscv = GridSearchCV(svm.SVC(),params,cv=10, verbose=3, scoring='accuracy', n_jobs=3 )

data_train, data_test = train_test_split(data,test_size=0.25,random_state=0)

y_train = data_train[:,2].astype(int)
X_train = data_train[:,:2]
y_test = data_test[:,2].astype(int)
X_test = data_test[:,:2]

gscv.fit(X_train,y_train)

y_pred = gscv.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot()
ax.legend(loc='upper right')
ax.set_title('(a)')
ax.scatter(X_train[:,0],X_train[:,1],c=y_train,s=2)
ax.legend().remove()
plt.savefig(sample_name+'_a.png',dpi=300)
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('(b)')
plot_decision_regions(X_train,y_train, clf=gscv,  res=0.01)
plt.savefig(sample_name+'_b.png',dpi=300)
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('(c)')
plot_decision_regions(X_test, y_test, clf=gscv,  res=0.01)
plt.savefig(sample_name+'_c.png',dpi=300)
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('(d)')
plot_decision_regions(X_test, y_pred, clf=gscv,  res=0.01)
plt.savefig(sample_name+'_d.png',dpi=300)
plt.show()
plt.close()


print('\nResult:')
print('\tBest Score: ', gscv.best_score_)
print('\tBest Params: ', gscv.best_params_ )
print('\tAccuracy (test): ',accuracy_score(y_test,y_pred))

