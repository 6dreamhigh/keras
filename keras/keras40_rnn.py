import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) #(10,)
#y가 없기 때문에 정해줘야 한다.
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])#훈련 시킬 y 마지막이 10이므로 10을 포함시키면 안된다.
y = np.array([4,5,6,7,8,9,10])
'''
데이터 형태
x    y
123 |4
234 |5
345 |6
456 |7
567 |8
678 |9
789 |10
'''
print(x.shape,y.shape) #(7, 3) (7,)
x = x.reshape(7,3,1)
print(x.shape) #(7,3,1) 3개씩 연산한것을 한개 

#2.모델
model = Sequential()
model.add(SimpleRNN(64,input_shape=(3,1))) #하나씩 가중치가 부여된다는 것 명시해야 함
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
#컴파일, 훈련
model.compile(loss ='mae',optimizer ='adam')
model.fit(x,y,epochs = 100)

#평가, 예측
loss = model.evaluate(x,y)
print(loss)