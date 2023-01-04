import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#1. 데이터

x= np.array([1,2,3,4,5,6])
y= np.array([1,2,3,5,4,6])


#2. 모델 구성 1->3->5->4->2->1:심층 신경망 구현-순서대로 진행되어 처음 input layer외에는 명시하지 않아도 된다.
#hidden layer 구성을 바꾸어도 (노드의 개수 / 레이어의 개수 변경)정확도가 나아질 수 있다.
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(35))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일 ,훈련
model.compile(loss='mae',optimizer ='adam')
model.fit(x,y,epochs=10, batch_size =3)

#4.평가 ,예측
result = model.predict([6])
print('6의 결과 : ',result)