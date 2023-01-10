from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])


#2.모델 구성
model = Sequential()
model.add(Dense(5, input_dim =1))#2*5 연산
model.add(Dense(4))#6*4
model.add(Dense(3))#5*3
model.add(Dense(2))#4*2
model.add(Dense(1))#3*1

model.summary()#아키텍처의 구조와 연산량(+bias값이 더해진 파라미터의 개수)을 나타낸다. 

