#36_1복붙
import numpy as np
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)#(60000, 28, 28) (60000,) 흑백 데이터
#(60000, 28,28, 1) 
print(x_test.shape,y_test.shape)#(60000, 28, 28) (60000,) 흑백 데이터
#(10000, 28, 28) (10000,)



#1. 데이터
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)
# print(x_train.shape)#(60000, 28, 28, 1)
# print(x_test.shape)#(10000, 28, 28, 1)


x_train, x_test = x_train / 255.0, x_test / 255.0

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten,MaxPooling2D, Dropout

model = Sequential()
model.add(Dense(128,activation = 'relu', input_shape = (28,28)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='linear'))
model.add(Dense(10, activation='softmax'))
model.summary()

          #원핫을 안해서 loss = Sparse_Categorical_crossentropy
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
                


#earlystopping, mcp 적용 / val 적용

# loss;  0.07674340158700943
# acc :  0.9772999882698059