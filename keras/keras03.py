import tensorflow as tf
import numpy as np

print(tf.__version__)

# 1. 데이터 준비
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1,input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mae',optimizer= 'adam')
model.fit(x,y,epochs=3000)

#4. 평가 ,예측
result = model.predict([6])
print('결과:',result)