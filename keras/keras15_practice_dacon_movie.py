import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1. 데이터 
path = './_data/movie/'
train_csv = pd.read_csv(path+'movies_train.csv',index_col=0)
test_csv = pd.read_csv(path+'movies_test.csv',index_col=0)
submission = pd.read_csv(path+'submission.csv',index_col=0)

print(train_csv)
print(train_csv.shape) #(1459, 10) input_dim = 9 ->count 제외시켜야 함
print(submission.shape)
print(train_csv.columns)

#결측치 처리
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)

train_csv.head(4)

train_csv = train.drop(['dir_prev_bfnum'],axis = 1)
test_csv =  test.drop(['dir_prev_bfnum'],axis = 1)