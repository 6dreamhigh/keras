import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
path = '../_save/'
#1.데이터 준비
samsung = pd.read_csv("C:/study/_data/stock/samsung.csv", header=0, index_col=None, sep=',', encoding='cp949', thousands=',').loc[::-1]
# print(samsung)
# print(samsung.shape)

amore = pd.read_csv("C:/study/_data/stock/amore.csv", header=0, index_col=None, sep=',', encoding='cp949', thousands=',').loc[::-1]
# print(amore)
# print(amore.shape)  

# 삼성전자 x ,y 데이터 추출
samsung_x = samsung[['고가', '저가','종가', '외인(수량)', '기관']]
samsung_y = samsung[['시가']].to_numpy() 

# print(samsung_x)
# print(samsung_y)
# print(samsung_x.shape) 
# print(samsung_y.shape) 

# 아모레 x, y 데이터 추출
amore_x = amore.loc[1979:0,['고가', '저가', '종가', '외인(수량)', '시가']]
# print(amore_x)
# print(amore_x.shape) 

samsung_x = MinMaxScaler().fit_transform(samsung_x)
amore_x = MinMaxScaler().fit_transform(amore_x)

def split_data(dataset, timesteps):
    tmp = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        tmp.append(subset)
    return np.array(tmp)

samsung_x = split_data(samsung_x, 5)
amore_x = split_data(amore_x, 5)
# print(samsung_x.shape)
# print(amore_x.shape) 

samsung_y = samsung_y[4:, :]

# predict data
samsung_x_predict = samsung_x[-1].reshape(-1, 5, 5)
amore_x_predict = amore_x[-1].reshape(-1, 5, 5)
# print(samsung_x_predict.shape)
# print(amore_x_predict.shape) 

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, amore_x_train, amore_x_test  = train_test_split(
    samsung_x, samsung_y, amore_x, train_size=0.8, random_state=123, shuffle=False)

print(samsung_x_train.shape, samsung_x_test.shape)  
print(samsung_y_train.shape, samsung_y_test.shape) 
print(amore_x_train.shape, amore_x_test.shape) 

#2. 모델 구성
# 삼성전자 모델
input_s = Input(shape=(5, 5))
dense_s1 = LSTM(128, return_sequences=True,activation='relu')(input_s)
dense_s2 = Dropout(0.1)(dense_s1)
dense_s3 = LSTM(256, activation='relu')(dense_s2)
dense_s4 = Dense(512, activation='relu')(dense_s3)
dense_s5 = Dropout(0.25)(dense_s4)
dense_s6 = Dense(128, activation='relu')(dense_s5)
dense_s7 = Dropout(0.15)(dense_s6)
dense_s8 = Dense(128, activation='relu')(dense_s7)
dense_s9 = Dropout(0.15)(dense_s8)
dense_s10 = Dense(84, activation='relu')(dense_s8)
dense_s11 = Dropout(0.2)(dense_s9)
dense_s12 = Dense(54, activation='relu')(dense_s10)
output_s = Dense(1)(dense_s11)

# 아모레퍼시픽 모델
input_a = Input(shape=(5, 5))
dense_a1 = LSTM(128, return_sequences=True,activation='relu')(input_a)
dense_a2 = Dropout(0.1)(dense_a1)
dense_a3 = LSTM(125, activation='relu')(dense_a2)
dense_a4 = Dense(512, activation='relu')(dense_a3)
dense_a5 = Dropout(0.1)(dense_a4)
dense_a6 = Dense(512, activation='relu')(dense_a5)
dense_a7 = Dropout(0.2)(dense_a6)
dense_a8 = Dense(128, activation='relu')(dense_a7)
dense_a9 = Dropout(0.15)(dense_a8)
dense_a10 = Dense(64, activation='relu')(dense_a9)
dense_a11 = Dropout(0.1)(dense_a10)
dense_a12 = Dense(40, activation='relu')(dense_a11)
output_a = Dense(1)(dense_a12)

# 병합
merge1 = concatenate([output_s, output_a])
merge2 = Dense(128, activation='relu')(merge1)
merge3 = Dense(256, activation='relu')(merge2)
merge4 = Dense(512, activation='relu')(merge3)
merge5 =Dropout(0.3)(merge4)
merge6 = Dense(128, activation='relu')(merge5)
merge7 = Dense(64, activation='relu')(merge6)
merge8 = Dense(32, activation='relu')(merge7)
output_mg = Dense(1, activation='relu')(merge8)

model = Model(inputs=[input_s, input_a], outputs=[output_mg])
model.summary()

#3.컴파일 , 훈련
model.compile(loss='mse', optimizer= 'adam')

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                              restore_best_weights=True,                        
                              verbose=1 
                              )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath = filepath + 'Samsung_st' + date + '_' + filename
                      )

#4.예측, 평가
model.fit([samsung_x_train, amore_x_train], samsung_y_train , epochs=500, batch_size=100, validation_split=0.2, callbacks=[es, mcp])

model.save_weights(path + 'stock_weight.h5') # 가중치만 저장

loss=model.evaluate([samsung_x_test, amore_x_test], samsung_y_test, batch_size=1024)
samsung_y_predict=model.predict([samsung_x_predict, amore_x_predict])

print("loss : ", loss)
print("삼성전자 시가 :" , samsung_y_predict)

# loss :  36938536.0
# 삼성전자 시가 : [[63471.418]]