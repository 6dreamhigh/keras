
import numpy as np
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)#(60000, 28, 28) (60000,) 흑백 데이터
#(60000, 28,28, 1) 
print(x_test.shape,y_test.shape)#(60000, 28, 28) (60000,) 흑백 데이터
#(10000, 28, 28) (10000,)



#1. 데이터
#이미지는 4차원 , 현재 3차원이므로 4차원으로 바꿔줘야 함
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape)#(60000, 28, 28, 1)
print(x_test.shape)#(10000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

x_train, x_test = x_train / 255.0, x_test / 255.0

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten,MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(filters=128, kernel_size = (3,3),
                 input_shape = (28,28,1), activation='relu'))#(27,27,128) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=64, kernel_size = (2,2)))#(26,26,64)
model.add(Dropout(0.15))
model.add(Conv2D(filters=64, kernel_size = (2,2)))#(25,25,64)
model.add(Flatten())#(40000,)
model.add(Dense(64, activation= 'relu')) #input_shape = (40000,)  -
model.add(Dense(10, activation='softmax'))
#model.summary()
          #원핫을 안해서 loss = Sparse_Categorical_crossentropy
          
#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy',optimizer='adam',
              metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
es  = EarlyStopping(monitor = 'val_loss',mode = 'min', 
                               patience = 10,
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
                
# loss;  0.044294435530900955
# acc :  0.9878000020980835

#earlystopping, mcp 적용 / val 적용


'''
1.filter
Conv2D 층에서 filter는 이미지에서 특징을 분리해내는 기능을 한다. 
filters의 값은 합성곱에 사용되는 필터의 개수이며, 출력 공간의 차원을 결정한다.

2.kernel_size
kernel_size 는 합성곱에 사용되는 필터의 크기이다.

3.input_shape
입력 데이터의 형태 (input_shape)는 아래와 같이 
MNIST 숫자 이미지 하나의 형태에 해당하는 (28, 28, 1)로 설정

4. pooling
합성곱에 의해 얻어진 특징맵으로부터 값을 샘플링해서 정보를 압축하는 과정
맥스풀링은 특정 영역에서 가장 큰 값을 샘플링하는 풀링 방식

5. strides
풀링 필터를 이동시키는 간격

'''