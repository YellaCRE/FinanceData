# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:27:37 2021

@author: noble
"""

import pandas as pd
data=pd.read_csv("C:/bank.csv")
data=data.drop(['Unnamed: 0'],axis=1)
from sklearn.preprocessing import MinMaxScaler
import imblearn
Y=data[['TARGET']]
X=data.drop(['TARGET'],axis=1)
X=MinMaxScaler().fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=1004)
from imblearn.under_sampling import RandomUnderSampler
X_undersampled, Y_undersampled = RandomUnderSampler(random_state=0).fit_resample(X_train,Y_train)
from imblearn.over_sampling import SMOTE
X_smote, Y_smote = SMOTE(random_state=0).fit_resample(X_train,Y_train)
from imblearn.over_sampling import ADASYN
X_ada, Y_ada = ADASYN(random_state=0).fit_sample(X_train,Y_train)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score
model=LogisticRegression()
model.fit(X_undersampled,Y_undersampled)
y_pred=model.predict(X_test)
uscore=model.score(X_test,Y_test)
uascore=accuracy_score(Y_test,y_pred)
urscore=recall_score(Y_test, y_pred)

model2=LogisticRegression()
model2.fit(X_smote,Y_smote)
y_pred2=model2.predict(X_test)
uscore2=model2.score(X_test,Y_test)
uascore2=accuracy_score(Y_test,y_pred2)
urscore2=recall_score(Y_test, y_pred2)

model3=LogisticRegression()
model3.fit(X_ada,Y_ada)
y_pred3=model3.predict(X_test)
uscore3=model3.score(X_test,Y_test)
uascore3=accuracy_score(Y_test,y_pred3)
urscore3=recall_score(Y_test, y_pred3)