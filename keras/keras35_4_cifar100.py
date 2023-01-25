from keras.datasets import cifar10
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape,y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 1)



print(np.unique(y_train, return_counts=True))

x_train, x_test = x_train / 255.0, x_test / 255.0 # 데이터 전처리
#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size = (2,2),
                 input_shape = (32,32,3), 
                 padding ='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size = (2,2)))
model.add(Dropout(0.15))
model.add(Conv2D(filters=64, kernel_size = (2,2)))
model.add(Flatten())
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(100, activation='softmax'))


#컴파일 , 훈련
model.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor ='val_loss',
                   mode = 'min',
                   patience = 3,
                   restore_best_weights=True,
                   verbose = 1)
date = datetime.datetime.now()
date = date.strftime("%m%d_%H %M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor ='val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True,
                      filepath = filepath+'k34_3_'+ date+ '_' + filename)
model.fit(x_train, y_train, epochs = 100,
          batch_size = 32,
          validation_split=0.2,
          callbacks = [es,mcp],
          verbose = 1)

#평가, 예측
result = model.evaluate(x_test,y_test)
print('loss; ',result[0])
print('acc : ',result[1])
                
# loss;  0.9979073405265808
# acc :  0.6570000052452087
#earlystopping, mcp 적용 / val 적용

