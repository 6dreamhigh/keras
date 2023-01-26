import numpy as np
import pandas as pd
import datetime
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint     
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path = '../_data/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

#train_csv = train_csv.interpolate(method='linear', limit_direction='forward')
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']



x_train, x_test, y_train, y_test = train_test_split(x, y,
   test_size=0.2, shuffle=True, random_state=123
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


test_csv = scaler.transform(test_csv)
print(x_train.shape, x_test.shape) #(8708, 8) (2178, 8)


x_train = x_train.reshape(8708, 8, 1, 1)
x_test = x_test.reshape(2178, 8, 1, 1)

x_train = tf.expand_dims(x_train, axis=-1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(8,1,1)))
model.add(Flatten()) #dropout 훈련 시에만 적용된다.
model.add(Dense(40, activation = 'relu'))

model.add(Dense(30, activation = 'relu'))

model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))



#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])                                                  
# 모델을 더 이상 학습을 못할 경우(loss, metric등의 개선이 없을 경우), 학습 도중 미리 학습을 종료시키는 콜백함수                                                                                            
es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'min', 
                   patience = 100, #참을성     
                   #restore_best_weights = False, 
                   verbose = 1)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")


filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf


mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k31_05_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=10, 
          batch_size=32, 
          validation_split=0.25, 
          callbacks=[es, mcp])

model.save(path + 'keras31_dropout05_save_model.h5')  #모델 저장 (가중치 포함 안됨)



#4. 예측, 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

#제출
#test_csv에 데이터를 넣어야되는데, 위에서 reshape으로 인하여 형태 변환이 이뤄진 상태여서 error 발생
# reshape를 통해서 원래 형태 변경을 통해 해결
y_submit = model.predict(test_csv.reshape(6493, 8, 1, 1))

submission['count'] = y_submit
submission.to_csv(path + 'submission_0106.csv')

print("===================================")
print(y_test)
print(y_predict)
print("submit : ", y_submit) 
print("R2 : " , r2)
print("RMSE : ", RMSE(y_test, y_predict))
print("===================================")

'''
===================================
loss :  [0.35212478041648865, 0.4382113218307495]
R2 :  0.7337006900174228
===================================
'''
