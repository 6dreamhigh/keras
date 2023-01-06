import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1.데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train = np.array(x[1:12])
y_train = np.array(y[1:12])
x_test = np.array(x[12:15])
y_test = np.array(y[12:15])
x_validation = np.array(x[15:18])
y_validation = np.array(y[15:18])

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

17의 예측값 :  [[15.89]]

'''