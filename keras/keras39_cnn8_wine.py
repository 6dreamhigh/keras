import numpy as np
import pandas as pd
import datetime

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #파이썬 클래스 대문자로 시작   
from sklearn.preprocessing import MinMaxScaler, StandardScaler



path = './_save/'

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, # False 의 문제점은 데이터 분할 시, 한쪽으로 쏠림 현상 발생으로 데이터의 훈련도의 오차가 심해진다. 
    random_state=123, # random_state 를 사용 시, 분리된 데이터가 비율이 안맞는 현상 발생
    test_size=0.3,
    stratify=y # 분리된 데이터가 비율이 일정하게 됨, 데이터 자체(y)가 분류형 일 때만 가능 , load_boston 데이터는 회귀형 데이터라 안됨.

)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) #(124, 13) (54, 13)

x_train = x_train.reshape(124, 13, 1, 1)
x_test = x_test.reshape(54, 13, 1, 1)


#2. 모델구성 # 분류형 모델
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(13, 1, 1)))
model.add(Conv2D(32, (2,1), input_shape=(13, 1, 1)))
model.add(Flatten()) #dropout 훈련 시에만 적용된다.
model.add(Dense(24, activation = 'relu'))
model.add(Dense(16, activation = 'linear'))
model.add(Dense(8, activation = 'linear'))
model.add(Dense(4, activation = 'linear'))
model.add(Dense(1, activation = 'softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


es = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=100, #참을성     
                              restore_best_weights=True, #최소값에 했던 지점에서 멈춤
                              verbose=1
                              )



date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf

# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k31_08_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=500, 
          batch_size=1, 
          validation_split=0.25, 
          callbacks=[es, mcp])

model.save(path + 'keras31_dropout08_save_model.h5')  #모델 저장 (가중치 포함 안됨)




#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)



y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1) # y_predict 가장 큰 값의 자릿수 뽑음 : 예측한 값

print( 'y_predict(예측값)' , y_predict)

y_test = np.argmax(y_test, axis=1) # y_test : y_test 값, (원 핫 인코딩을 진행했기 때문에, 다시 원복) 

#print( 'y_test(원래값)' , y_test)

acc = accuracy_score(y_test, y_predict) 

print(acc)

'''
loss :  0.13963143527507782
accuracy :  0.9259259104728699
y_predict(예측값) [2 1 1 2 2 2 0 2 0 1 1 0 1 0 0 0 1 0 0 2 0 1 0 2 0 0 1 1 0 1 1 2 0 0 2 1 2
 2 0 0 1 2 0 1 0 2 1 1 1 2 0 2 1 1]
0.9259259259259259
'''