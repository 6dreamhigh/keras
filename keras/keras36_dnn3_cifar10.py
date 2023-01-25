from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)


# print(x_train.shape)
# print(x_test.shape)
print(np.unique(y_train, return_counts=True))

x_train, x_test = x_train / 255.0, x_test / 255.0 # 데이터 전처리

#2. 모델

model = Sequential()
model.add(Dense(128,activation = 'relu', input_shape = (3072,)))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='linear'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy',optimizer='adam',
              metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
es  = EarlyStopping(monitor = 'val_loss',mode = 'min', 
                               patience = 7,
                               restore_best_weights=True, 
                               verbose = 1)
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H %M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor ='val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True,
                      filepath = filepath+'k34_'+ date+ '_' + filename)

model.fit(x_train, y_train, epochs = 100,
          batch_size = 100,
          validation_split=0.2,
          callbacks = [es,mcp],
          verbose = 1)

#평가, 예측
result = model.evaluate(x_test,y_test)
print('loss; ',result[0])
print('acc : ',result[1])