import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = np.loadtxt('sample2.csv',delimiter=',')

params = [
        {'C': np.logspace(-2,4,7,base=10), 'kernel':['linear']},
        {'C': np.logspace(-2,4,7,base=10), 'gamma':np.logspace(-5,1,7,base=10), 'kernel':['rbf']},
        {'C': np.logspace(-2,4,7,base=10), 'degree':[2,3,4], 'gamma': np.logspace(-5,1,7,base=10), 'kernel':['poly']},
        {'C': np.logspace(-2,4,7,base=10), 'gamma':np.logspace(-5,1,7,base=10),'kernel':['sigmoid']}
    ]

gscv = GridSearchCV(svm.SVC(),params,cv=5, verbose=3, scoring='accuracy', n_jobs=7 )

data_train, data_test = train_test_split(data,test_size=0.25,random_state=0)

y_train = data_train[:,2]
X_train = data_train[:,:2]
y_test = data_test[:,2]
X_test = data_test[:,:2]

gscv.fit(X_train,y_train)

print( 'best params: ', gscv.best_params_ )
y_pred = gscv.predict(X_test)
print(classification_report(y_test, y_pred))
