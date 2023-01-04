import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10]) #(10,)
y = np.array(range(10))
x_train = x[:-3]
x_test = x[-3:]
y_train = y[:7]
y_test = y[7:]



print(x_train,x_test,y_train,y_test)
'''
#2.모델 구성
model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train, epochs = 1110,batch_size =1)

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
result = model.predict([11])
print('result : ',result)

'''