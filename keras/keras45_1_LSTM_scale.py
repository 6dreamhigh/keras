import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM 
#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],
             [10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,0,60,70])
x_predict = np.array([50,60,70])
print(x.shape,y.shape)  #(13, 3) (13,)


#2. 모델
model = Sequential()
model.add(LSTM(units = 600,input_shape = (3,1)))
model.add(Dense(512,activation='relu'))
model.add(Dense(308,activation='relu'))
model.add(Dense(264,activation='relu'))
model.add(Dense(132,activation='relu'))
model.add(Dense(56,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.summary()


#컴파일, 훈련
model.compile(loss ='mae',optimizer ='adam')
model.fit(x,y,epochs = 1450)

#평가, 예측
loss = model.evaluate(x,y)
print('loss : ',loss)
#:Model was constructed with shape (None, 3, 1) 
y_pred = np.array([50,60,70]).reshape(1,3,1)
result = model.predict(y_pred)
print('[50,60,70]의 결과 :',result)


# loss :  0.7281479239463806
# [50,60,70]의 결과 : [[76.8076]]