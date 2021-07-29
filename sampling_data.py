# Smote_NC test
from imblearn.over_sampling import SMOTENC
import pandas as pd

# 데이터 불러오기 # 기존 데이터 1열 삭제
df = pd.read_csv('C:/Users/HCJ/Desktop/2021_Summer/Finance_data/FinanceData.csv')

# 데이터 분석
# print(df.describe())

# 타겟,설명변수 나누기
Y_list = ['TARGET']
df_X = df.drop(Y_list, axis=1)
df_Y = df.loc[:, Y_list]

# print(df_X)
# print(df_Y)
print(f'Original dataset shape {df_X.shape}')

# SmoteNC 적용
df_indx = [0, 91] # 이부분 하드코딩이니까 
sm = SMOTENC(categorical_features=df_indx, random_state=42)
X_res, Y_res = sm.fit_resample(df_X, df_Y)

# 샘플링 후 데이터 확인
Y_Cnt = Y_res.groupby('TARGET').size()
print(Y_Cnt)

# 타켓,설명변수 합치기 및 csv로 변환
result = pd.concat([Y_res, X_res], axis=1)
print(result)
result.to_csv("sampling.csv")

