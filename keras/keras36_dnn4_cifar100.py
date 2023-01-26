import numpy as np
import datetime


from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #파이썬 클래스 대문자로 시작   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dropout

path = './_save/'


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data() #교육용 자료, 이미 train/test 분류

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)(훈련)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1) (테스트)

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],dtype=int64))

x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3)   


x_train = x_train/255.
x_test = x_test/255.


print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)(훈련)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1) (테스트)


# 2. 모델 구성 
model = Sequential()
model.add(Dense(1024, input_shape=(32,32, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='linear'))
model.add(Dense(100, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae', 'mse'])


es = EarlyStopping(monitor='val_loss', 
                              mode='auto', 
                              patience=10,
                              restore_best_weights=True, 
                              verbose=1
                              )


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf

# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'min',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k34_3_cifer100' + date + '_'+ filename)



model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.2, callbacks=[es, mcp])

model.save(path + 'keras34_3_cifer100.h5')  #모델 저장 (가중치 포함 안됨)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ',  result[0]) # loss
print('acc : ',  result[1]) # acc



'''
[dnn]
loss :  0.008052281104028225
acc :  0.3352999985218048
[cnn]

loss :  1.4575377702713013
acc :  0.48730000853538513
'''