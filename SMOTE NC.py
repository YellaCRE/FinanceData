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
#더미코딩
nominal=X_smotenc.columns[[42,43,44,63,64,65,66,67]]
X_smotenc_dummy = pd.get_dummies(X_smotenc,columns=nominal)
X_test_dummy=pd.get_dummies(X_test,columns=nominal)

#로지스틱
#smotenc
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_smotenc,Y_smotenc)
Y_pred_log=log.predict(X_test)
#성능평가
from sklearn import metrics
log_accuracy=metrics.accuracy_score(Y_test,Y_pred_log)
log_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_log)
#dummy
log1=LogisticRegression()
log1.fit(X_smotenc_dummy,Y_smotenc)
Y_pred_logd=log1.predict(X_test_dummy)
#성능평가
logd_accuracy=metrics.accuracy_score(Y_test,Y_pred_logd)
logd_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_logd)

#DT
#smotenc
from sklearn.tree import DecisionTreeClassifier
t1=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
t1.fit(X_smotenc,Y_smotenc)
Y_pred_dt=t1.predict(X_test)
#성능평가
tree_accuracy=metrics.accuracy_score(Y_test,Y_pred_dt)
tree_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_dt)
#dummy
t2=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
t2.fit(X_smotenc_dummy,Y_smotenc)
Y_pred_dtd=t2.predict(X_test_dummy)
#성능평가
treed_accuracy=metrics.accuracy_score(Y_test,Y_pred_dtd)
treed_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_dtd)

#NaiveBayes
#smotenc
from sklearn.naive_bayes import CategoricalNB
catNB1=CategoricalNB()
catNB1.fit(X_smotenc,Y_smotenc)
Y_pred_nb=catNB1.predict(X_test)
#성능평가
naive_accuracy=metrics.accuracy_score(Y_test,Y_pred_nb)
naive_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_nb)
#dummy
catNB2=CategoricalNB()
catNB2.fit(X_smotenc_dummy,Y_smotenc)
Y_pred_nbd=catNB2.predict(X_test_dummy)
#성능평가
naived_accuracy=metrics.accuracy_score(Y_test,Y_pred_nbd)
naived_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_nbd)

#RandomForest
#smotenc
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20, max_depth=5,random_state=0)
clf.fit(X_smotenc,Y_smotenc)
Y_pred_rf=clf.predict(X_test)
#성능평가
forest_accuracy=metrics.accuracy_score(Y_test,Y_pred_rf)
forest_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_rf)
#dummy
clf2 = RandomForestClassifier(n_estimators=20, max_depth=5,random_state=0)
clf2.fit(X_smotenc_dummy,Y_smotenc)
Y_pred_rfd=clf2.predict(X_test_dummy)
#성능평가
forestd_accuracy=metrics.accuracy_score(Y_test,Y_pred_rfd)
forestd_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_rfd)

#MLP
#smotenc
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd', alpha=0.01, batch_size=32,learning_rate_init=0.1, max_iter=500)  
mlp.fit(X_smotenc, Y_smotenc)
Y_pred_mlp = mlp.predict(X_test)
#성능 평가
nc_mlp_accuracy=metrics.accuracy_score(Y_test,Y_pred_mlp)
nc_mlp_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_mlp)
#dummy
mlp2 = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd', alpha=0.01, batch_size=32,learning_rate_init=0.1, max_iter=500)  
mlp2.fit(X_smotenc_dummy, Y_smotenc)
Y_pred_mlpd = mlp2.predict(X_test_dummy)
#성능 평가
nc_mlpd_accuracy=metrics.accuracy_score(Y_test,Y_pred_mlpd)
nc_mlpd_score=metrics.precision_recall_fscore_support(Y_test,Y_pred_mlpd)

