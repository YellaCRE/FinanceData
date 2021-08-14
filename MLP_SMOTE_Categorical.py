# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:39:40 2021

@author: noble
"""

#데이터 전처리
import pandas as pd
data=pd.read_csv("C:/bank.csv")
for col_name in data.columns:
    data[col_name]=pd.Categorical(data[col_name])
Y=data[['TARGET']]
X=data.drop(['Unnamed: 0','TARGET'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=1004)
from imblearn.over_sampling import SMOTEN
smoten = SMOTEN()
X_train_smote, Y_train_smote = smoten.fit_resample(X_train,Y_train)

#MLP 샘플링 X
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd', alpha=0.01, batch_size=32,learning_rate_init=0.1, max_iter=500)  
mlp.fit(X_train, Y_train)
Y_pred = mlp.predict(X_test)
#성능 평가
from sklearn import metrics
mlp_accuracy=metrics.accuracy_score(Y_test,Y_pred)
mlp_score=metrics.precision_recall_fscore_support(Y_test,Y_pred)

#MLP 샘플링 O
mlp2 = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd', alpha=0.01, batch_size=32,learning_rate_init=0.1, max_iter=500)  
mlp2.fit(X_train_smote, Y_train_smote)
Y_pred2 = mlp2.predict(X_test)
#성능 평가
smote_mlp_accuracy=metrics.accuracy_score(Y_test,Y_pred2)
smote_mlp_score=metrics.precision_recall_fscore_support(Y_test,Y_pred2)

#MLP 샘플링 O, RELU
mlp3 = MLPClassifier(activation='relu')  
mlp3.fit(X_train_smote, Y_train_smote)
Y_pred3 = mlp3.predict(X_test)
#성능 평가
smote_mlpr_accuracy=metrics.accuracy_score(Y_test,Y_pred3)
smote_mlpr_score=metrics.precision_recall_fscore_support(Y_test,Y_pred3)