#53-3복붙 fit 추가
import numpy as np


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# #1. 데이터
# np.save('../_data/brain/brain_x_train.npy', arr=xy_train[0][0])
# np.save('../_data/brain/brain_y_train.npy', arr=xy_train[0][1])
# # np.save('../_data/brain/brain_xy_train.npy', arr=xy_train[0])
# np.save('../_data/brain/brain_x_test.npy', arr=xy_test[0][0])
# np.save('../_data/brain/brain_y_test.npy', arr=xy_test[0][1])

x_train = np.load('../_data/brain/brain_x_train.npy')
y_train = np.load('../_data/brain/brain_y_train.npy')
x_test = np.load('../_data/brain/brain_x_test.npy')
y_test = np.load('../_data/brain/brain_y_test.npy')


print(x_train.shape, x_test.shape) #(160, 200, 200, 1) (120, 200, 200, 1)
print(y_train.shape, y_test.shape) #(160,) (120,)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(256, (3,3), input_shape=(200,200,1), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

hist = model.fit(x_train, y_train, batch_size=16, epochs=50, validation_split=0.2)

#4. 평가, 예측
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1]) #loss[100]
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])
# results=model.evaluate(xy_test)
# print('loss, acc : ', results)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) #판사이즈(figsize(가로길이, 세로길이) 단위는 inch이다.)
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid() #격자
plt.xlabel('epochs') #x축 라벨
plt.ylabel('loss')   #y축 라벨
plt.title('ImageDataGenerator2') #제목 이름 표시
plt.legend() #선의 이름
#plt.legend(loc='upper left') 
plt.show()