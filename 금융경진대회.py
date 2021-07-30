import pandas as pd
import imblearn
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
import matplotlib.pyplot as plt

# 데이터 불러오기
data=pd.read_csv("C:/bank.csv")

# 데이터 전처리
Y=data[['TARGET']]
X=data.drop(['Unnamed: 0','TARGET'],axis=1)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=1004)

# 데이터 샘플링

X_undersampled, Y_undersampled = RandomUnderSampler(random_state=0).fit_resample(X_train,Y_train)
X_smote, Y_smote = SMOTE(random_state=0).fit_resample(X_train,Y_train)
X_ada, Y_ada = ADASYN(random_state=0).fit_sample(X_train,Y_train)

# DT 모델 형성 및 결과
# undersampling
t1=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
t1.fit(X_undersampled,Y_undersampled)
y_pred=t1.predict(X_test)

under_naive_accuracy=metrics.accuracy_score(Y_test,y_pred)
under_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred)
print("Undersampling DT","\n"+"Accuracy : ", under_naive_accuracy, "\n"+"Precision, Recall, Fscore : " under_naive_score)

# Smote
t2=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
t2.fit(X_smote,Y_smote)
y_pred1=t2.predict(X_test)

smote_naive_accuracy=metrics.accuracy_score(Y_test,y_pred1)
smote_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred1)
print("Smote DT","\n"+"Accuracy : ", smote_naive_accuracy, "\n"+"Precision, Recall, Fscore : " smote_naive_score)
# Adasyn
t3=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
t3.fit(X_ada,Y_ada)
y_pred2=t3.predict(X_test)

ada_naive_accuracy=metrics.accuracy_score(Y_test,y_pred2)
ada_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred2)
print("Adasyn DT","\n"+"Accuracy : ", ada_naive_accuracy, "\n"+"Precision, Recall, Fscore : " ada_naive_score)

# NB 모델 형성 및 결과
# undersamling
multiNB1=MultinomialNB()
multiNB1.fit(X_undersampled,Y_undersampled)
y_pred3=multiNB1.predict(X_test)

under_naive_accuracy=metrics.accuracy_score(Y_test,y_pred3)
under_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred3)
print("Undersampling NB","\n"+"Accuracy : ", under_naive_accuracy, "\n"+"Precision, Recall, Fscore : " under_naive_score)

# Smote
multiNB2=MultinomialNB()
multiNB2.fit(X_smote,Y_smote)
y_pred4=multiNB2.predict(X_test)

smote_naive_accuracy=metrics.accuracy_score(Y_test,y_pred4)
smote_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred4)
print("Smote NB","\n"+"Accuracy : ", smote_naive_accuracy, "\n"+"Precision, Recall, Fscore : " smote_naive_score)

# Adasyn
multiNB3=MultinomialNB()
multiNB3.fit(X_ada,Y_ada)
y_pred5=multiNB3.predict(X_test)

ada_naive_accuracy=metrics.accuracy_score(Y_test,y_pred5)
ada_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred5)
print("Adasyn NB","\n"+"Accuracy : ", ada_naive_accuracy, "\n"+"Precision, Recall, Fscore : " ada_naive_score)
