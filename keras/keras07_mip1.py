import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터 준비
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape) #(2,10) 데이터.shape->행렬의 구조

print(y.shape) #(10,)

x = x.T # x의 전치행렬 
print(x.shape) #(10,2)

#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim = 2)) 
#input_dim은 열(feature,특성,column) 의 개수 의미.행 무시 열 우선!!!2개가 들어가서 5개가 결과로 나옴
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mae',optimizer = 'adam')
model.fit(x , y, epochs = 100, batch_size = 1)

#4.평가, 예측
loss = model.evaluate(x,y)
print('loss : ',loss)

result = model.predict([[10,1.4]])#20가까운 값이 나와야 함
print('[10,1.4]의 예측값 : ',result)


'''
결과 정리: 가장 좋은 loss값 적기

'''

