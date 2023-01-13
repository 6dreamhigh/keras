from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler as MMS, StandardScaler as SDS

import numpy as np

path = '../_save/'
# path = 'c:/study/_save/'
# model.save(path + 'keras29_1_save_model.h5')
# model.save('./_save_keras29_1_saver_model.h5')

#1. 데이터
dataset = load_boston()         # 보스턴 집 값에 대한 데이터
x = dataset.data                # 방 넓이, 방 개수 등 → 독립변수
y = dataset.target              # 집 값 → 종속변수

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=3333
)

# scaler = MMS()
scaler = SDS()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성 (순차형)
model = Sequential()
model.add(Dense(5,input_shape=(13,)))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(70))
model.add(Dense(1))
model.summary()
#평가 시에도 드랍아웃이 적용되는가? X ,드랍아웃은 훈련 시에만 적용되고, 평가시에는 전체 다 사용된다.

# # (함수형) Total params: 52,225
# input = Input(shape=(13,))
# dense1 = Dense(32)(input)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(64, activation= 'relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(256, activation= 'relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(128, activation= 'relu')(drop3)
# output = Dense(1)(dense4)
# model = Model(inputs=input, outputs=output)
# model.summary()







# #3. 컴파일 , 훈련
# model.compile(loss = 'mse', optimizer='adam',metrics = ['mae'])

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es= EarlyStopping(monitor='val_loss',patience = 20,
#                               mode='min',
#                               restore_best_weights=True,
#                               verbose = 1)
# import datetime
# date = datetime.datetime.now()#시간 기록 
# print(date)# 현재 시간이 나온다. 2023-01-12 15:02:36.903303
# print(type(date))#<class 'datetime.datetime'>-> 파일에 집어넣을려면 string타입이어야 함
# date = date.strftime("%m%d_%H %M") #시간을 string으로 바꿔주는 역할을 한다. 
# print(date)#0112_15 02
# print(type(date))#<class 'str'>
# '''
# 시간 순서대로 가중치가 저장되어 가장 최적의 가중치인 경우 파일로 저장하여 확인할 수 있게 한다. 
# '''

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #0037-0.0048.hdf5 ->시간과 성능이 있는 hdf5파일 의미




# mcp = ModelCheckpoint(monitor = 'val_loss',mode = 'auto',
#                       verbose = 1,
#                       save_best_only=True,
#                       filepath = filepath+'k31_01_'+ date+ '_' + filename)

# model.fit(x_train, y_train, epochs=10000, 
#           batch_size=16, validation_split=0.2,
#           callbacks = [es,mcp],
#           verbose=1)

# #model.save(path+ "keras30_ModelcheckPoint3_save_model.h5")
# # model = load_model(path+'MCP/keras30_ModelCheckPoint1.hdf5')

# #4. 평가 및 예측
# print("==================================1. 기본 출력 ============================")
# model.save(path+ "keras30_ModelcheckPoint3_save_model.h5")
# mse,mae = model.evaluate(x_test,y_test)
# print("mse : ",mse)
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print("R2: ", r2)

