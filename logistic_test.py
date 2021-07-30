import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTENC
import pandas as pd


# 데이터 불러오기
data = pd.read_csv('C:/Users/HCJ/Desktop/2021_Summer/Finance_data/FinanceData.csv')
fin = data.drop(['Unnamed: 0'], axis=1)

# 데이터 분석
print(fin.groupby('TARGET').size())
print(fin['TARGET'].value_counts(normalize = True))

# 데이터 전처리
Y_list = ['TARGET']
fin_X = fin.drop(Y_list, axis=1)
fin_Y = fin.loc[:, Y_list]

x_train, x_test, y_train, y_test = train_test_split(fin_X, fin_Y, train_size=0.8, test_size=0.2)

# SmoteNC 적용

sm = SMOTENC(random_state=42)
X_samp, Y_samp = sm.fit_resample(x_train, y_train)

# 샘플링 후 데이터 확인
print(Y_samp['TARGET'].value_counts()/y_train['TARGET'].shape[0]*100)

# 모델 형성
model = LogisticRegression()
model.fit(x_samp, y_samp)

# 검증 및 성능지표
y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
score = metrics.precision_recall_fscore_support(y_test, y_pred)

print("Accuracy : ", accuracy, "Precision, Recall, Fscore : " score)

x_train, x_test, y_train, y_test = train_test_split(fin_samp_X, fin_samp_Y, train_size=0.8, test_size=0.2)

# 모델 형성 및 검증
log = LogisticRegression()
log.fit(x_train, y_train)

print('\n', "Sampling R2 : ", log.score(x_train, y_train))


log.fit(x_train, y_train)
y_pred = log.predict(x_test)
R = log.score(x_test, y_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
score = metrics.precision_recall_fscore_support(y_test, y_pred)

print(accuracy, score)
