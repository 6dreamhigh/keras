import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
'''
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle = True,
    random_state=123
)
ratio=(0.625,0.185,0.185)
7:3:3
'''
#1.데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.65,
    shuffle = True,
    random_state=123
)
x_test,x_val, y_test, y_val = train_test_split(
    x_test, y_test,
    test_size=0.5,
    shuffle = True,
    random_state=123
)
print('x_train :' ,x_train)
print('y_train : ',y_train)
print('x_test :' ,x_test)
print('y_test : ',y_test)
print('x_val :' ,x_val)
print('y_val : ',y_val)
'''
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train = np.array(x[1:11])
y_train = np.array(y[1:11])
x_test = np.array(x[11:14])
y_test = np.array(y[10:14])
x_val = np.array(x[14:17])
y_val = np.array(y[14:17])
'''
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
          validation_data=(x_val,y_val))

#4.평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

result = model.predict([17])
print('17의 예측값 : ',result)