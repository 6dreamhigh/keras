import numpy as np
import pandas as pd
import tensorflow as tf
import datetime

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint     

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score



path = './_save/'

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets['target']


# One-hot Encoding 방법
# 1. keras 메서드 활용
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, 
    random_state=123,
    test_size=0.3,
    stratify=y
)

print(x_train.shape, x_test.shape) #(105, 4) (45, 4)

x_train = x_train.reshape(105, 2, 2, 1)
x_test = x_test.reshape(45, 2, 2, 1)


#2. 모델구성 
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(2, 2, 1)))
model.add(Conv2D(32, (2,2), input_shape=(2, 2, 1)))
model.add(Flatten()) #dropout 훈련 시에만 적용된다.
model.add(Dense(24, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'linear'))
model.add(Dense(4, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                  

es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'min', 
                   patience = 20, #참을성     
                   #restore_best_weights = False, 
                   verbose = 1)



date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf


# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k31_07_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=500, 
          batch_size=8, 
          validation_split=0.25, 
          callbacks=[es, mcp])

model.save(path + 'keras31_dropout07_save_model.h5')  #모델 저장 (가중치 포함 안됨)



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)


y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1) # y_predict 가장 큰 값의 자릿수 뽑음 : 예측한 값

print( 'y_predict(예측값)' , y_predict)

y_test = np.argmax(y_test, axis=1) 

print( 'y_test(원래값)' , y_test)

acc = accuracy_score(y_test, y_predict) 

print(acc)

'''
[dnn]
loss :  0.02484283410012722
accuracy :  0.057244524359703064
y_predict(예측값) [2 0 1 1 0 2 2 2 2 2 0 2 0 2 1 2 0 0 2 2 2 2 2 1 2 1 0 2 1 0 0 0 0 1 1 1 0
 1 0 2 0 1 1 0 1]
y_test(원래값) [1 0 1 1 0 2 2 1 2 2 0 2 0 2 1 2 0 0 2 2 2 2 2 1 2 1 0 2 1 0 0 0 0 1 1 1 0
 1 0 2 0 1 1 0 1]
0.9555555555555556
[cnn]
loss :  0.2222222238779068
accuracy :  0.4444483816623688
y_predict(예측값) [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0]
y_test(원래값) [1 0 1 1 0 2 2 1 2 2 0 2 0 2 1 2 0 0 2 2 2 2 2 1 2 1 0 2 1 0 0 0 0 1 1 1 0
 1 0 2 0 1 1 0 1]
0.3333333333333333
'''