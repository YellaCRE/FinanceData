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
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

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
DT1 = DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
DT1.fit(X_undersampled,Y_undersampled)
y_pred1=DT1.predict(X_test)

under_DT_accuracy=metrics.accuracy_score(Y_test,y_pred1)
under_DT_score=metrics.precision_recall_fscore_support(Y_test,y_pred1)
print("Undersampling DT","\n"+"Accuracy : ", under_DT_accuracy, "\n"+"Precision, Recall, Fscore : " under_DT_score)

# Smote
DT2=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
DT2.fit(X_smote,Y_smote)
y_pred2=DT2.predict(X_test)

smote_DT_accuracy=metrics.accuracy_score(Y_test,y_pred2)
smote_DT_score=metrics.precision_recall_fscore_support(Y_test,y_pred2)
print("Smote DT","\n"+"Accuracy : ", smote_DT_accuracy, "\n"+"Precision, Recall, Fscore : " smote_DT_score)
# Adasyn
DT3=DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
DT3.fit(X_ada,Y_ada)
y_pred3=DT3.predict(X_test)

ada_DT_accuracy=metrics.accuracy_score(Y_test,y_pred3)
ada_DT_score=metrics.precision_recall_fscore_support(Y_test,y_pred3)
print("Adasyn DT","\n"+"Accuracy : ", ada_DT_accuracy, "\n"+"Precision, Recall, Fscore : " ada_DT_score)

# NB 모델 형성 및 결과
# undersamling
multiNB1=MultinomialNB()
multiNB1.fit(X_undersampled,Y_undersampled)
y_pred4=multiNB1.predict(X_test)

under_naive_accuracy=metrics.accuracy_score(Y_test,y_pred4)
under_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred4)
print("Undersampling NB","\n"+"Accuracy : ", under_naive_accuracy, "\n"+"Precision, Recall, Fscore : " under_naive_score)

# Smote
multiNB2=MultinomialNB()
multiNB2.fit(X_smote,Y_smote)
y_pred5=multiNB2.predict(X_test)

smote_naive_accuracy=metrics.accuracy_score(Y_test,y_pred5)
smote_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred5)
print("Smote NB","\n"+"Accuracy : ", smote_naive_accuracy, "\n"+"Precision, Recall, Fscore : " smote_naive_score)

# Adasyn
multiNB3=MultinomialNB()
multiNB3.fit(X_ada,Y_ada)
y_pred6=multiNB3.predict(X_test)

ada_naive_accuracy=metrics.accuracy_score(Y_test,y_pred6)
ada_naive_score=metrics.precision_recall_fscore_support(Y_test,y_pred6)
print("Adasyn NB","\n"+"Accuracy : ", ada_naive_accuracy, "\n"+"Precision, Recall, Fscore : " ada_naive_score)

# logistic 모델 형성 및 결과
# optimization
'''
Y = data[['TARGET']]
X = data.drop(['TARGET'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1004

X = MinMaxScaler().fit_transform(X) 
# 뭔가 이상함. 테스트데이터 스플릿 이전에 optimization 해야하는 것 같은데 스플릿 이후에 optimization 하는게 필요없는 코드 같음
'''
# under sampling
Logi1 = LogisticRegression()
Logi1.fit(X_undersampled, Y_undersampled)
y_pred7 = Logi1.predict(X_test)

under_Logi_accuracy=metrics.accuracy_score(Y_test,y_pred7)
under_Logi_score=metrics.precision_recall_fscore_support(Y_test,y_pred7)
print("Undersampling Logi","\n"+"Accuracy : ", under_Logi_accuracy, "\n"+"Precision, Recall, Fscore : " under_Logi_score)

# Smote
Logi2 = LogisticRegression()
Logi2.fit(X_smote, Y_smote)
y_pred8 = Logi2.predict(X_test)

smote_Logi_accuracy=metrics.accuracy_score(Y_test,y_pred8)
smote_Logi_score=metrics.precision_recall_fscore_support(Y_test,y_pred8)
print("Smote Logi","\n"+"Accuracy : ", smote_Logi_accuracy, "\n"+"Precision, Recall, Fscore : " smote_Logi_score)

# Adasyn
Logi3 = LogisticRegression()
Logi3.fit(X_ada, Y_ada)
y_pred9 = Logi3.predict(X_test)

ada_Logi_accuracy=metrics.accuracy_score(Y_test,y_pred9)
ada_Logi_score=metrics.precision_recall_fscore_support(Y_test,y_pred9)
print("Adasyn Logi","\n"+"Accuracy : ", ada_Logi_accuracy, "\n"+"Precision, Recall, Fscore : " ada_Logi_score)
