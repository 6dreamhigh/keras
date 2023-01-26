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
model = Sequential()                     #(N,3,1)
model.add(LSTM(units = 600,input_shape = (3,1),
               return_sequences= True)) #  (None, 3, 600)  
model.add(LSTM(300))  # return_sequences설정하지 않고 하면LSTM으로부터 3차원을 받아야 되는데 2차원을 받아 오류뜸
# expected ndim=3, found ndim=2. Full shape received: (None, 600)--->Reshape layer / return_sequences= True
model.add(Dense(512,activation='relu'))
model.add(Dense(308,activation='relu'))
model.add(Dense(264,activation='relu'))
model.add(Dense(132,activation='relu'))
model.add(Dense(56,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
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
x_pred = np.array([50,60,70]).reshape(1,3,1)
result = model.predict(x_pred)
print('[50,60,70]의 결과 :',result)


# # loss :  0.40743082761764526
# # [50,60,70]의 결과 : [[74.775475]]