import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #파이썬 클래스 대문자로 시작   
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import tensorflow as tf


path = './_save/'


# 1. 데이터 
datasets = fetch_covtype()

x = datasets.data
y = datasets['target']

y = y.reshape(581012, 1) 


ohe = OneHotEncoder() 

y = ohe.fit_transform(y) 
y = y.toarray()


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, 
    random_state=333, 
    test_size=0.3,
    stratify=y 
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) #(406708, 54) (174304, 54)


x_train = x_train.reshape(406708, 6, 3, 3)
x_test = x_test.reshape(174304, 6, 3, 3)


#2. 모델구성
model = Sequential()
model.add(Conv2D(1024, (2,1), input_shape=(6, 3, 3)))#kernel size가 직사각형도 가능하다.
#model.add(MaxPooling2D())
model.add(Conv2D(512, (2,1)))
#model.add(MaxPooling2D())
model.add(Conv2D(256, (2,1)))
model.add(Flatten()) 
model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))
model.add(Dense(24, activation = 'linear'))
model.add(Dense(7, activation = 'softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


es = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=2,
                              restore_best_weights=True, 
                              verbose=1
                              )


import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
print(date)   #0112_1502
print(type(date)) #<class 'str'>


filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf

# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k31_10_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=5000, 
          batch_size=128, 
          validation_split=0.25, 
          callbacks=[es, mcp])

#model.save(path + 'keras31_dropout10_save_model.h5')  #모델 저장 (가중치 포함 안됨)




#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)



y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)

print( 'y_predict(예측값)' , y_predict)

y_test = np.argmax(y_test, axis=1)

print( 'y_test(원래값)' , y_test)

acc = accuracy_score(y_test, y_predict) 

print(acc)





'''
[dnn 방식]
loss :  0.6085023283958435
accuracy :  0.7451693415641785
y_predict(예측값) [1 0 2 ... 1 1 1]
y_test(원래값) [1 1 5 ... 1 4 1]
0.7451693592803378
[cnn 방식]
accuracy :  0.7189335823059082
y_predict(예측값) [1 0 2 ... 1 1 1]
y_test(원래값) [1 1 5 ... 1 4 1]
0.7189335872957592
[cnn 방식 + 튜닝 후]
loss :  0.31951621174812317
accuracy :  0.8671745657920837
y_predict(예측값) [1 1 2 ... 1 1 0]
y_test(원래값) [1 1 5 ... 1 4 1]
0.8671745915182669
'''