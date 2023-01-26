#31_1 복붙
import numpy as np 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint     
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


path = './_save/'


# 1. 데이터
dataset = load_boston() 

x = dataset.data
y = dataset.target

# 123, 365, 1, 100000
x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=1
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape) #(404, 13) (102, 13)

x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)

print(x_train.shape, x_test.shape) #(404, 13) (102, 13)

#2. 모델구성(순차형)
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(13, 1, 1)))
model.add(Flatten()) #dropout 훈련 시에만 적용된다.
model.add(Dense(40, activation = 'linear'))

model.add(Dense(30, activation = 'linear'))

model.add(Dense(20, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                  
# 모델을 더 이상 학습을 못할 경우(loss, metric등의 개선이 없을 경우), 학습 도중 미리 학습을 종료시키는 콜백함수                                                                                            
es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'min', 
                   patience = 20, #참을성     
                   #restore_best_weights = False, 
                   verbose = 1)

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
                      filepath = filepath + 'k30_01_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=5000, 
          batch_size=8, 
          validation_split=0.25, 
          callbacks=[es, mcp])

model.save(path + 'keras30_ModelCheckPoint3_save_model.h5')  #모델 저장 (가중치 포함 안됨)




#4. 평가, 예측

print('========================1. 기본 출력============================')
mse, mae = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('1. mse : ' , mse)
print('1. R2_스코어 : ', r2)


print('========================2. load_model 출력============================')
model2 = load_model(path + 'keras30_ModelCheckPoint3_save_model.h5')
mse, mae = model2.evaluate(x_test, y_test)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('2. mse : ' , mse)
print('2. R2_스코어 : ', r2)


print('========================3. ModelCheckPoint 출력============================')
#가장 좋은 지점을 가지고만 사용했기 때문에 제일 성능 좋게 나온다.
#선생님은 CheckPoint 사용
model3 = load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')
mse, mae = model3.evaluate(x_test, y_test)
print('3. mse : ' , mse)

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('3. R2_스코어 : ', r2)


'''
========================1. 기본 출력============================
4/4 [==============================] - 0s 590us/step - loss: 9.1758 - mae: 2.3048
1. mse :  9.175808906555176
1. R2_스코어 :  0.9071531927190991
========================2. load_model 출력============================
4/4 [==============================] - 0s 6ms/step - loss: 9.1758 - mae: 2.3048
2. mse :  9.175808906555176
2. R2_스코어 :  0.9071531927190991
========================3. ModelCheckPoint 출력============================
4/4 [==============================] - 0s 0s/step - loss: 8.4508 - mae: 2.2247
3. mse :  8.450822830200195
3. R2_스코어 :  0.9144890652804591
'''