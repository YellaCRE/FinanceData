# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 19:33:14 2021

@author: noble
"""
#데이터 전처리
import pandas as pd
data=pd.read_csv("C:/bank.csv")
Y=data[['TARGET']]
X=data.drop(['Unnamed: 0','TARGET'],axis=1)
X.iloc[:,[42,43,44,63,64,65,66,67]] = X.iloc[:,[42,43,44,63,64,65,66,67]].astype('category')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=1004)
from imblearn.over_sampling import SMOTENC
smote_nc = SMOTENC(categorical_features=[42,43,44,63,64,65,66,67], random_state=0)
X_smotenc, Y_smotenc = smote_nc.fit_resample(X_train, Y_train)

#로지스틱
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_smotenc,Y_smotenc)
Y_pred=log.predict(X_test)
#성능평가
from sklearn import metrics
log_accuracy=metrics.accuracy_score(Y_test,Y_pred)
log_score=metrics.precision_recall_fscore_support(Y_test,Y_pred)

#DT
from sklearn.tree import DecisionTreeClassifier
t1=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
t1.fit(X_smotenc,Y_smotenc)
Y_pred2=t1.predict(X_test)
#성능평가
tree_accuracy=metrics.accuracy_score(Y_test,Y_pred2)
tree_score=metrics.precision_recall_fscore_support(Y_test,Y_pred2)

#NaiveBayes
from sklearn.naive_bayes import CategoricalNB
catNB1=CategoricalNB()
catNB1.fit(X_smotenc,Y_smotenc)
Y_pred3=catNB1.predict(X_test)
#성능평가
naive_accuracy=metrics.accuracy_score(Y_test,Y_pred3)
naive_score=metrics.precision_recall_fscore_support(Y_test,Y_pred3)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20, max_depth=5,random_state=0)
clf.fit(X_smotenc,Y_smotenc)
Y_pred4=clf.predict(X_test)
#성능평가
forest_accuracy=metrics.accuracy_score(Y_test,Y_pred4)
forest_score=metrics.precision_recall_fscore_support(Y_test,Y_pred4)\

#MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd', alpha=0.01, batch_size=32,learning_rate_init=0.1, max_iter=500)  
mlp.fit(X_smotenc, Y_smotenc)
Y_pred5 = mlp.predict(X_test)
#성능 평가
nc_mlp_accuracy=metrics.accuracy_score(Y_test,Y_pred5)
nc_mlp_score=metrics.precision_recall_fscore_support(Y_test,Y_pred5)

