import numpy as np
import datetime

from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #파이썬 클래스 대문자로 시작   


path = './_save/'

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #교육용 자료, 이미 train/test 분류

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) reshape (훈련)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,) (테스트)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28, 1) (10000,)
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))


print(np.unique(y_train, return_counts=True))


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input

# 2. 모델 구성 
inputs = Input(
                 shape=(28, 28, 1)
               )
hidden1 = MaxPooling2D() (inputs)
hidden2 = Conv2D(filters=64, kernel_size=(2, 2), padding='same', strides=2) (hidden1)
hidden3 = Conv2D(filters=64, kernel_size=(2, 2)) (hidden2)
hidden4 = Flatten() (hidden3)
output = Dense(10, activation='softmax') (hidden4)


model = Model(inputs=inputs, outputs=output)

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])


es = EarlyStopping(monitor='val_loss', 
                              mode='min', 
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
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k34_1_mnist' + date + '_'+ filename)



model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.25,callbacks=[es, mcp])

model.save(path + 'keras34_1_minist.h5')  #모델 저장 (가중치 포함 안됨)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ',  result[0]) # loss
print('acc : ',  result[1]) # acc


# es, mcp 적용 / val 적용

'''
[maxpool 적용 전]
Total params: 1,543,914
[maxpool 적용 후]
Total params: 397,034
loss :  0.07935530692338943
acc :  0.9801999926567078
'''