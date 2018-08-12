# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:33:58 2017

@author: user
"""

# Importing the libraries
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#numerical describe
dataset.describe()

#catgary describe
dataset.describe(include=['O'])


dataset[['NumOfProducts', 'Exited']].groupby(['NumOfProducts'], \
       as_index=False).mean().sort_values(by='Exited', ascending=False)

dataset[['HasCrCard', 'Exited']].groupby(['HasCrCard'], \
       as_index=False).mean().sort_values(by='Exited', ascending=False)

dataset[['IsActiveMember', 'Exited']].groupby(['IsActiveMember'], \
       as_index=False).mean().sort_values(by='Exited', ascending=False)

dataset[['Gender', 'Exited']].groupby(['Gender'], \
       as_index=False).mean().sort_values(by='Exited', ascending=False)

dataset[['Geography', 'Exited']].groupby(['Geography'], \
       as_index=False).mean().sort_values(by='Exited', ascending=False)


g = sns.FacetGrid(dataset, col='Exited')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(dataset, col='Exited', row='Gender', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

grid = sns.FacetGrid(dataset, col='Exited', row='Geography', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


dataset = dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


dataset['Gender'] = dataset['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)


dataset.isnull().sum()

dataset['Geography'] = dataset['Geography'].map( {'Spain': 0, 'France': 1, 'Germany':2} ).astype(int)


dataset['AgeBand'] = pd.cut(dataset['Age'], 5)
dataset[['AgeBand', 'Exited']].groupby(['AgeBand'], 
        as_index=False).mean().sort_values(by='AgeBand', ascending=True)


dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 32), 'Age'] = 0
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 1
dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 63), 'Age'] = 2
dataset.loc[(dataset['Age'] > 63) & (dataset['Age'] <= 73), 'Age'] = 3
dataset.loc[(dataset['Age'] > 73) & (dataset['Age'] <= 92), 'Age'] = 4

dataset = dataset.drop(['AgeBand'], axis=1)



dataset['CreditB'] = pd.cut(dataset['CreditScore'], 9)
dataset[['CreditB', 'Exited']].groupby(['CreditB'], 
        as_index=False).mean().sort_values(by='CreditB', ascending=True)

dataset.loc[ dataset['CreditScore'] <= 349.5, 'CreditScore'] = 0
dataset.loc[(dataset['CreditScore'] > 349.5) & (dataset['CreditScore'] <= 405.556), 'CreditScore'] = 1
dataset.loc[(dataset['CreditScore'] > 405.556) & (dataset['CreditScore'] <= 461.111), 'CreditScore'] = 2
dataset.loc[(dataset['CreditScore'] > 461.111) & (dataset['CreditScore'] <= 516.667), 'CreditScore'] = 3
dataset.loc[(dataset['CreditScore'] > 516.667) & (dataset['CreditScore'] <= 572.222), 'CreditScore'] = 4
dataset.loc[(dataset['CreditScore'] > 572.222) & (dataset['CreditScore'] <= 627.778), 'CreditScore'] = 5
dataset.loc[(dataset['CreditScore'] > 627.778) & (dataset['CreditScore'] <= 683.333), 'CreditScore'] = 6
dataset.loc[(dataset['CreditScore'] > 683.333) & (dataset['CreditScore'] <= 738.889), 'CreditScore'] = 7
dataset.loc[(dataset['CreditScore'] > 738.889) & (dataset['CreditScore'] <= 794.444), 'CreditScore'] = 8
dataset.loc[(dataset['CreditScore'] > 794.444) & (dataset['CreditScore'] <= 850.0), 'CreditScore'] = 9
dataset.loc[(dataset['CreditScore'] > 850.0), 'CreditScore' ] = 10

dataset = dataset.drop(['CreditB'], axis=1)


dataset['BalanceB'] = pd.cut(dataset['Balance'], 7)
dataset[['BalanceB', 'Exited']].groupby(['BalanceB'], 
        as_index=False).mean().sort_values(by='BalanceB', ascending=True)

dataset.loc[ dataset['Balance'] <= -250.898, 'Balance'] = 0
dataset.loc[(dataset['Balance'] > -250.898) & (dataset['Balance'] <= 35842.584), 'Balance'] = 1
dataset.loc[(dataset['Balance'] > 35842.584) & (dataset['Balance'] <= 71685.169), 'Balance'] = 2
dataset.loc[(dataset['Balance'] > 71685.169) & (dataset['Balance'] <= 107527.753), 'Balance'] = 3
dataset.loc[(dataset['Balance'] > 107527.753) & (dataset['Balance'] <= 143370.337), 'Balance'] = 4
dataset.loc[(dataset['Balance'] > 143370.337) & (dataset['Balance'] <= 179212.921), 'Balance'] = 5
dataset.loc[(dataset['Balance'] > 179212.921) & (dataset['Balance'] <= 215055.506), 'Balance'] = 6
dataset.loc[(dataset['Balance'] > 215055.506) & (dataset['Balance'] <= 250898.09), 'Balance'] = 7
dataset.loc[(dataset['Balance'] > 250898.09), 'Balance' ] = 8

dataset = dataset.drop(['BalanceB'], axis=1)


dataset['SalaryB'] = pd.cut(dataset['EstimatedSalary'], 3)
dataset[['SalaryB', 'Exited']].groupby(['SalaryB'], 
        as_index=False).mean().sort_values(by='SalaryB', ascending=True)

dataset.loc[ dataset['EstimatedSalary'] <= -188.401, 'EstimatedSalary'] = 0
dataset.loc[(dataset['EstimatedSalary'] > -188.401) & (dataset['EstimatedSalary'] <= 66671.88), 'EstimatedSalary'] = 1
dataset.loc[(dataset['EstimatedSalary'] > 66671.88) & (dataset['EstimatedSalary'] <= 133332.18), 'EstimatedSalary'] = 2
dataset.loc[(dataset['EstimatedSalary'] > 133332.18) & (dataset['EstimatedSalary'] <= 199992.48), 'EstimatedSalary'] = 3
dataset.loc[(dataset['EstimatedSalary'] > 199992.48), 'EstimatedSalary' ] = 4

dataset = dataset.drop(['SalaryB'], axis=1)

X = dataset.drop("Exited", axis=1)
Y = dataset["Exited"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





X = X_train.iloc[: , :].values

x = X_test.iloc[:,:].values


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Logistic Regression

logreg = LogisticRegression(penalty = 'l2', C = 0.1)
logreg.fit(X, y_train)
Y_pred = logreg.predict(x)
acc_log = round(logreg.score(X, y_train) * 100, 2)
acc_log


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# Support Vector Machines

svc = SVC()
svc.fit(X, y_train)
Y_pred = svc.predict(x)
acc_svc = round(svc.score(X, y_train) * 100, 2)
acc_svc


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, y_train)
Y_pred = knn.predict(x)
acc_knn = round(knn.score(X, y_train) * 100, 2)
acc_knn


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X, y_train)
Y_pred = gaussian.predict(x)
acc_gaussian = round(gaussian.score(X, y_train) * 100, 2)
acc_gaussian


# Perceptron

perceptron = Perceptron()
perceptron.fit(X, y_train)
Y_pred = perceptron.predict(x)
acc_perceptron = round(perceptron.score(X, y_train) * 100, 2)
acc_perceptron


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X, y_train)
Y_pred = linear_svc.predict(x)
acc_linear_svc = round(linear_svc.score(X, y_train) * 100, 2)
acc_linear_svc


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X, y_train)
Y_pred = sgd.predict(x)
acc_sgd = round(sgd.score(X, y_train) * 100, 2)
acc_sgd


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, y_train)
Y_pred = decision_tree.predict(x)
acc_decision_tree = round(decision_tree.score(X, y_train) * 100, 2)
acc_decision_tree



# Random Forest

random_forest = RandomForestClassifier(n_estimators=1000, criterion="entropy", n_jobs = -1)
random_forest.fit(X, y_train)
Y_pred = random_forest.predict(x)
random_forest.score(X, y_train)
acc_random_forest = round(random_forest.score(X, y_train) * 100, 2)
acc_random_forest




#compare
    models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
                  'Random Forest', 'Naive Bayes', 'Perceptron', 
                  'Stochastic Gradient Decent', 'Linear SVC', 
                  'Decision Tree'], 
        'Score': [acc_svc, acc_knn, acc_log, 
                  acc_random_forest, acc_gaussian, acc_perceptron, 
                  acc_sgd, acc_linear_svc, acc_decision_tree]})
    models.sort_values(by='Score', ascending=False)
    
    
    from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
