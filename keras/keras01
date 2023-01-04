#문제 x = [1,2,3] , y = [1,2,3]일 경우 x =4인 경우 예측하기
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터 준비
x = np.array([1,2,3])
y = np.array([1,2,3])

#2.모델 준비하기
model = Sequential()
model.add(Dense(1,input_dim = 1))

#3.컴파일, 훈련
model.compile(loss ='mae', optimizer = 'adam')
model.fit(x,y,epochs =2000)

result = model.predict([4])
print("result : ",result)