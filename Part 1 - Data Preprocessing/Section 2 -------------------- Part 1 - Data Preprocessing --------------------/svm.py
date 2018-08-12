#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:13:47 2017

@author: Vincent
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%pylab


# Importing the dataset
dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#check null
dataset.isnull().sum()
test.isnull().sum()
dataset['Embarked'] = dataset['Embarked'].fillna('Unknown')


#class resort
class_map = {
        3 : 1, 
        2 : 2,
        1 : 3}
dataset['Pclass'] = dataset['Pclass'].map(class_map)
test['Pclass'] = test['Pclass'].map(class_map)


#into array
X = dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y = dataset.iloc[:, 1].values
t_x = test.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values


#encode
from sklearn.preprocessing import LabelEncoder
'''X[:,0] = LabelEncoder().fit_transform(X[:,0])'''
X[:,1] = LabelEncoder().fit_transform(X[:,1])
X[:,6] = LabelEncoder().fit_transform(X[:,6])

t_x[:,1] = LabelEncoder().fit_transform(t_x[:,1])
t_x[:,6] = LabelEncoder().fit_transform(t_x[:,6])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

imputer_t = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_t = imputer_t.fit(t_x)
t_x = imputer_t.transform(t_x)

#onehotencoder
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1: ] #onehotencoder trap

onehotencoder_t = OneHotEncoder(categorical_features = [6])
t_x = onehotencoder_t.fit_transform(t_x).toarray()



from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C = 0.01, random_state=0)
svm.fit(X, y)
plot_


from sklearn.cross_validation import cross_val_score
cv_scores = np.mean(cross_val_score(svm, X, y, scoring='roc_auc', cv=5))
print (cv_scores)



from matplotlib.colors import ListedColormap

def plot_decision_region(X, y, classifier, resolution=0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #plot the dicision surface
    x1_min, x1_max = X[:, 4].min() - 1, X[:, 4].max() + 1
    x2_min, x2_max = X[:, 8].min() - 1, X[:, 8].max() + 1
    
    xx1, xx2 = np.meshgrid(np,arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx1, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot class sample
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y ==cl, 1],
                    alpha= 0.8, c=cmap(idx),
                    marker = markers[idx], label=cl)
        
        
        

plot_decision_region(X, y, classifier = svm)
plt.xlabel()
plt.ylabel()
plt.legend(loc='upper left' )
plt.show()