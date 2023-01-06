import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1.데이터
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_validation = np.array([14,15,16])
y_validation = np.array([14,15,16])

#2. 모델
model = Sequential()
model.add(Dense(5,input_dim = 1))
model.add(Dense(3,activation='relu'))
model.add(Dense(52, activation= 'relu'))
model.add(Dense(38, activation= 'sigmoid'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(42, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(1))

#3.컴파일 ,훈련
model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x_train,y_train,epochs=2000,batch_size =1,
          validation_data=(x_validation,y_validation))

#4.평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

result = model.predict([17])
print('17의 예측값 : ',result)

'''
loss :  0.24196751415729523
17의 예측값 :  [[14.538305]]

'''