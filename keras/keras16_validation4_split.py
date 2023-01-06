import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1.데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train,x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    shuffle = True,
    random_state=123
)
print('x_train :' ,x_train)
print('y_train : ',y_train)
print('x_test :' ,x_test)
print('y_test : ',y_test)
#print(x_train.shape)

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
          validation_split=0.25)

#4.평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

result = model.predict([17])
print('17의 예측값 : ',result)

#평가 기준은 loss가 아닌 val_loss로 잡아야 한다. 
'''
loss :  0.17616456747055054
17의 예측값 :  [[17.06819]]


'''