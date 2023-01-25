from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# print(x_train.shape,y_train.shape)#(60000, 28, 28) (60000,) 
# #(60000, 28,28, 1) 
# print(x_test.shape,y_test.shape)
# #(10000, 28, 28) (10000,)
# import matplotlib.pyplot as plt
# plt.imshow(x_train[10],'gray')
# plt.show()

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

x_train, x_test = x_train / 255.0, x_test / 255.0

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten,MaxPooling2D, Dropout


model = Sequential()
model.add(Conv2D(filters=128, kernel_size = (4,4),
                 input_shape = (28,28,1),
                 padding = 'same',
                 strides= 1,
                 activation='relu'))#(28,28,128) 
# model.add(MaxPooling2D((3, 3)))#연산은 없고 특성만 강화시킴

model.add(Conv2D(filters=64,
                 kernel_size = (4,4),
                 padding = 'same'))#(28,28,64)
model.add(MaxPooling2D((3, 3)))#연산은 없고 특성만 강화시킴
model.add(Dropout(0.15))
model.add(Conv2D(filters=64, kernel_size = (2,2)))#(25,25,64)
#model.add(MaxPooling2D((3,3)))
model.add(Flatten())#(40000,)
model.add(Dense(128, activation= 'relu')) 
model.add(Dropout(0.15))
model.add(Dense(10, activation='softmax'))
model.summary()
          #원핫을 안해서 loss = Sparse_Categorical_crossentropy
          
#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy',optimizer='adam',
              metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
es  = EarlyStopping(monitor = 'val_loss',mode = 'min', 
                               patience = 6,
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

# loss;  0.22912199795246124
# acc :  0.9190999865531921

