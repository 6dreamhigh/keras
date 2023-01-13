import numpy as np 

from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터
dataset = load_boston() 

x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=333
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성(함수형)
input1 = Input(shape=(13, ))
dense1 = Dense(50, activation='relu') (input1)
dense2 = Dense(40, activation='sigmoid') (dense1)
dense3 = Dense(30, activation='relu') (dense2)
dense4 = Dense(20, activation='linear') (dense3)
output1 = Dense(1, activation='linear') (dense4)

model = Model(inputs=input1, outputs=output1)
model.summary() #Total params: 4,611

path = './_save/'
# path = '../_save/'
# path = 'C:/study/_save/' #절대경로

model.save(path + 'keras29_1_save_model.h5')  #모델 저장 (가중치 포함 안됨)
#model.save('./_save/keras29_1_save_model.h5')

