# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:59:41 2021

@author: noble
"""

#데이터 전처리
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
import pandas as pd
data=pd.read_csv("C:/bank.csv")
Y=data[['TARGET']]
X=data.drop(['Unnamed: 0','TARGET'],axis=1)
X.iloc[:,[42,43,44,63,64,65,66,67]] = X.iloc[:,[42,43,44,63,64,65,66,67]].astype('category')
A=X.iloc[:,[42,43,44,63,64,65,66,67]]
B=X.drop(X.columns[[42,43,44,63,64,65,66,67]],axis=1)
sparse_features = A.columns
dense_features = B.columns
fixlen_feature_columns = [SparseFeat(feat, X[feat].nunique()+1,embedding_dim=4)
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=1004)
train_model_input = {name:X_train[name] for name in feature_names}
test_model_input = {name:X_test[name] for name in feature_names}
from imblearn.over_sampling import SMOTENC
smote_nc = SMOTENC(categorical_features=[42,43,44,63,64,65,66,67], random_state=0)
X_smotenc, Y_smotenc = smote_nc.fit_resample(X_train, Y_train)
train_model_input_smotenc = {name:X_smotenc[name] for name in feature_names}

#model fit
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, Y_train,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)