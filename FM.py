# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:05:24 2021

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

#FM, 샘플링 X
import xlearn as xl
import numpy as np
# DMatrix transition, if use field ,use must pass field map(an array) of features
xdm_train = xl.DMatrix(X_train, Y_train)
xdm_test = xl.DMatrix(X_test, Y_test)
# Training task
fm_model = xl.create_fm()  # Use factorization machine
# we use the same API for train from file
# that is, you can also pass xl.DMatrix for this API now
fm_model.setTrain(xdm_train)    # Training data
fm_model.setValidate(xdm_test)  # Validation data
# param:
#  0. regression task
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: mae
param = {'task':'binary', 'lr':0.2,
         'lambda':0.002, 'metric':'acc'}
# Start to train
# The trained model will be stored in model.out
fm_model.fit(param, './model_dm.out')
# Prediction task
# we use the same API for test from file
# that is, you can also pass xl.DMatrix for this API now
fm_model.setTest(xdm_test)  # Test data
fm_model.setSign() # Convert output to 0,1
# Start to predict
# The output result will be stored in output.txt
# if no result out path setted, we return res as numpy.ndarray
Y_pred = fm_model.predict("./model_dm.out")
#성능 평가
from sklearn import metrics
fm_accuracy=metrics.accuracy_score(Y_test,Y_pred)
fm_score=metrics.precision_recall_fscore_support(Y_test,Y_pred)

#FM, 샘플링 O
# DMatrix transition, if use field ,use must pass field map(an array) of features
xdm_train1 = xl.DMatrix(X_smotenc, Y_smotenc)
# Training task
fm_model1 = xl.create_fm()  # Use factorization machine
# we use the same API for train from file
# that is, you can also pass xl.DMatrix for this API now
fm_model1.setTrain(xdm_train1)    # Training data
fm_model1.setValidate(xdm_test)  # Validation data
# param:
#  0. regression task
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: mae
param = {'task':'binary', 'lr':0.03,
         'lambda':0.0001, 'metric':'acc'}
# Start to train
# The trained model will be stored in model.out
fm_model1.fit(param, './modeldm.out')
# Prediction task
# we use the same API for test from file
# that is, you can also pass xl.DMatrix for this API now
fm_model1.setTest(xdm_test)  # Test data
fm_model1.setSign() # Convert output to 0,1
# Start to predict
# The output result will be stored in output.txt
# if no result out path setted, we return res as numpy.ndarray
Y_pred1 = fm_model.predict("./modeldm.out")
#성능 평가
from sklearn import metrics
fm_smote_accuracy=metrics.accuracy_score(Y_test,Y_pred1)
fm_smote_score=metrics.precision_recall_fscore_support(Y_test,Y_pred1)