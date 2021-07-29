# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:46:00 2021

@author: noble
"""
import pandas as pd
data=pd.read_csv("C:/bank.csv")
import imblearn
Y=data[['TARGET']]
X=data.drop(['Unnamed: 0','TARGET'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=1004)
from imblearn.under_sampling import RandomUnderSampler
X_undersampled, Y_undersampled = RandomUnderSampler(random_state=0).fit_resample(X_train,Y_train)
from imblearn.over_sampling import SMOTE
X_smote, Y_smote = SMOTE(random_state=0).fit_resample(X_train,Y_train)
from imblearn.over_sampling import ADASYN
X_ada, Y_ada = ADASYN(random_state=0).fit_sample(X_train,Y_train)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score
t1=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
t1.fit(X_undersampled,Y_undersampled)
y_pred=t1.predict(X_test)
under_accuracy=accuracy_score(Y_test,y_pred)
under_recall=recall_score(Y_test,y_pred,average=None)
under_precision=precision_score(Y_test,y_pred,average=None)
t2=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
t2.fit(X_smote,Y_smote)
y_pred1=t2.predict(X_test)
smote_accuracy=accuracy_score(Y_test,y_pred1)
smote_recall=recall_score(Y_test,y_pred1,average=None)
smote_precision=precision_score(Y_test,y_pred1,average=None)
t3=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
t3.fit(X_ada,Y_ada)
y_pred2=t3.predict(X_test)
ada_accuracy=accuracy_score(Y_test,y_pred2)
ada_recall=recall_score(Y_test,y_pred2,average=None)
ada_precision=precision_score(Y_test,y_pred2,average=None)

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
import matplotlib.pyplot as plt
multiNB1=MultinomialNB()
multiNB1.fit(X_undersampled,Y_undersampled)
y_pred3=multiNB1.predict(X_test)
under_naive_accuracy=metrics.accuracy_score(Y_test,y_pred3)
under_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred3)
multiNB2=MultinomialNB()
multiNB2.fit(X_smote,Y_smote)
y_pred4=multiNB2.predict(X_test)
smote_naive_accuracy=metrics.accuracy_score(Y_test,y_pred4)
smote_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred4)
multiNB3=MultinomialNB()
multiNB3.fit(X_ada,Y_ada)
y_pred5=multiNB3.predict(X_test)
ada_naive_accuracy=metrics.accuracy_score(Y_test,y_pred5)
ada_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred5)





