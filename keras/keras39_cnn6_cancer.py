import numpy as np 
import datetime
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


path = './_save/'

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']


x_train, x_test, y_train, y_test = train_test_split (
    x, y, shuffle=True, random_state=123, test_size=0.2
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(455, 30) (114, 30)\


x_train = x_train.reshape(455, 6, 5, 1)
x_test = x_test.reshape(114, 6, 5, 1)

#2. 모델 구성 # 이진 모델
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(6, 5, 1)))
model.add(Conv2D(64, (2,2)))
model.add(Flatten()) #dropout 훈련 시에만 적용된다.
model.add(Dense(32, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(16, activation = 'linear'))
model.add(Dense(8, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))


model.compile(loss='binary_crossentropy', #이진데이터 이용 시 loss 함수는 binary_crossentropy 사용
              optimizer='adam',
              metrics=['accuracy'] # 이진 데이터 사용 시 metrics 함수의 accuracy 사용, metrics를 사용하면 히스토리에 나옴.
              )                                 


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
                      filepath = filepath + 'k31_06_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=500, 
          batch_size=8, 
          validation_split=0.25, 
          callbacks=[es, mcp])

model.save(path + 'keras31_dropout06_save_model.h5')  #모델 저장 (가중치 포함 안됨)




#3. 평가, 예측
#loss = model.evaluate(x_test, y_test)
loss, accuracy = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)


# 자료형 변환
y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)





print('============================================')
print('loss : ', loss, ' accuracy : ', accuracy )
print('============================================')
print(' accuracy_score : ', acc )
print('============================================')





'''
============================================
loss :  0.17042416334152222  accuracy :  0.9385964870452881
============================================
 accuracy_score :  0.9385964912280702
============================================
'''