# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


""" Data Engineering """


traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')


def featuresengi(inputdata):
    #Name
    inputdata['Title'] = inputdata.Name.str.extract('([A-Za-z]+)\.', expand=False)
    
    inputdata['Title'] = inputdata['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Sir', 'Lady', 'Capt', 'Countess', 'Don'], 'Others')
    inputdata['Title'] = inputdata['Title'].replace('Mlle', 'Miss')
    inputdata['Title'] = inputdata['Title'].replace('Ms', 'Miss')
    inputdata['Title'] = inputdata['Title'].replace('Mme', 'Mrs')
    Titledic = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,'Rev': 5, 'Others':6}
    inputdata['Title'] = inputdata['Title'].map(Titledic)
    inputdata['Title'] = inputdata['Title'].fillna(0)
    
    #Sibsp & Parch
    inputdata['Familysize'] = inputdata['SibSp'] + inputdata['Parch'] + 1
    inputdata['Familygroup'] = 0
    inputdata.loc[inputdata['Familysize'] == 1, 'Familygroup'] = 1
    inputdata.loc[inputdata['Familysize'] == 2, 'Familygroup'] = 2
    inputdata.loc[inputdata['Familysize'] == 3, 'Familygroup'] = 2
    inputdata.loc[inputdata['Familysize'] == 4, 'Familygroup'] = 3
    inputdata.loc[inputdata['Familysize'] == 5, 'Familygroup'] = 1
    inputdata.loc[inputdata['Familysize'] == 6, 'Familygroup'] = 1
    inputdata.loc[inputdata['Familysize'] == 7, 'Familygroup'] = 1
    
    #Sex  
    inputdata['Sex'] = inputdata['Sex'].map({'male':0, 'female':1}).astype(int)
    
    #Age
    for i in range(7):
        Titleage = inputdata[(inputdata['Title'] == i)]['Age'].dropna()
        Estiage = Titleage.median()
    for i in range(7):
        inputdata.loc[(inputdata.Age.isnull()) & (inputdata['Title'] == i), 'Age'] = Estiage
    inputdata['Age'] = inputdata['Age'].astype(int)
        
    inputdata.loc[ inputdata['Age'] <= 16, 'Age'] = 0
    inputdata.loc[(inputdata['Age'] > 16) & (inputdata['Age'] <= 30), 'Age'] = 1
    inputdata.loc[(inputdata['Age'] > 30) & (inputdata['Age'] <= 48), 'Age'] = 2
    inputdata.loc[(inputdata['Age'] > 48) & (inputdata['Age'] <= 63), 'Age'] = 3
    inputdata.loc[ inputdata['Age'] > 63, 'Age'] = 4
    
    #Fare    
    for i in range(1,4):
        Pfare = inputdata[(inputdata['Pclass'] == i)]['Fare'].dropna()
        Estifare = Pfare.median()
        inputdata.loc[(inputdata.Fare.isnull()) & (inputdata['Pclass'] == i), 'Fare'] = Estifare
    
    inputdata.loc[inputdata['Fare'] <= 10.4625, 'Fare'] = 0
    inputdata.loc[(inputdata['Fare'] > 7.91) & (inputdata['Fare'] <= 10.4625), 'Fare'] = 1
    inputdata.loc[(inputdata['Fare'] > 15.1) & (inputdata['Fare'] <= 29.7), 'Fare']   = 2
    inputdata.loc[inputdata['Fare'] > 29.7, 'Fare'] = 3
    inputdata['Fare'] = inputdata['Fare'].astype(int)
    
    #Pclass
    inputdata.loc[(inputdata['Sex'] == 1), 'Pclass'] += 3
    inputdata['Pclass'] = inputdata['Pclass'].astype(int)

featuresengi(traindata)
featuresengi(testdata)


""" Prediction """


testX  = testdata.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Familysize', 'Cabin', 'Embarked'], axis=1)
trainX = traindata[['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'Familygroup']]
trainY = traindata['Survived']

#searchrange = {'max_features':[1,2,3,4,5,6,'auto','log2','sqrt']}
#gsearch = GridSearchCV(RandomForestClassifier(n_estimators=220, min_samples_leaf=2, min_samples_split=5, criterion='entropy', max_depth=7, n_jobs = -1, oob_score=True), param_grid = searchrange, scoring='roc_auc',cv=5)
#gsearch.fit(trainX, trainY)
#print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)

randomforest = RandomForestClassifier(n_estimators=220, min_samples_leaf=2, min_samples_split=5, criterion='entropy', max_depth=7, n_jobs = -1, oob_score=True)
randomforest.fit(trainX, trainY)
print(randomforest.oob_score_)
testY = randomforest.predict(testX)

submission = pd.DataFrame({'PassengerId': testdata['PassengerId'], 'Survived': testY})
submission.to_csv('submission.csv', index=False)