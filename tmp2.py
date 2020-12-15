import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

run_name = 'hoo'
data = np.loadtxt('sampleB.csv',delimiter=',')

params = [
        {'C': np.logspace(-2,4,7,base=10), 'kernel':['linear']}#,
        #{'C': np.logspace(-2,4,1000,base=10), 'kernel':['linear']}#,
        #{'C': np.linspace(473,475,10), 'kernel':['linear']}#,
        #{'C': np.logspace(-2,4,13,base=10), 'gamma':np.logspace(-5,1,13,base=10), 'kernel':['rbf']},
        #{'C': np.logspace(-2,4,13,base=10), 'degree':[2,3], 'gamma': np.logspace(-5,1,13,base=10), 'kernel':['poly']}
        #{'C': np.linspace(14000,16000,10), 'degree':[2], 'gamma': np.linspace(1.0,1.5,10), 'kernel':['poly']}
        #{'C': np.logspace(-2,4,7,base=10), 'degree':[3], 'gamma': np.logspace(-5,1,7,base=10), 'kernel':['poly']}
        #{'C': np.logspace(-2,4,7,base=10), 'degree':[2], 'gamma': np.logspace(-5,1,7,base=10), 'coef0':np.linspace(-10,10,10),'kernel':['poly']}
        #{'C': np.logspace(3,4,1000,base=10), 'gamma':np.logspace(-3,-2,1000,base=10), 'kernel':['rbf']}#,
        #{'C': np.linspace(4150,4200,100), 'gamma':np.linspace(0.0090,0.0091,100), 'kernel':['rbf']}#,
    ]

gscv = GridSearchCV(svm.SVC(),params,cv=5, verbose=3, scoring='accuracy', n_jobs=8 )

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
plt.savefig(run_name+'_a.png',dpi=300)
plt.close()

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('(b)')
plot_decision_regions(X_train,y_train, clf=gscv,  res=0.01)
plt.savefig(run_name+'_b.png',dpi=300)
plt.close()

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('(c)')
plot_decision_regions(X_test, y_test, clf=gscv,  res=0.01)
plt.savefig(run_name+'_c.png',dpi=300)
plt.close()

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('(d)')
plot_decision_regions(X_test, y_pred, clf=gscv,  res=0.01)
plt.savefig(run_name+'_d.png',dpi=300)
plt.close()


print('\nResult:')
print('\tBest Score: ', gscv.best_score_)
print('\tBest Params: ', gscv.best_params_ )
print('\tAccuracy(test):',accuracy_score(y_test,y_pred))

