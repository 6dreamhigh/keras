import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#1. 데이터

#이미지 데이터를 증폭시켜 훈련시킴
train_datagen = ImageDataGenerator(
    rescale=1./255,         #크기 재조절 인수. 디폴트 값은 None
    horizontal_flip=True,   #인풋을 무작위로 가로로 뒤집습니다.
    vertical_flip=True,     #인풋을 무작위로 세로로 뒤집습니다.
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,       #무작위 회전의 각도 범위
    zoom_range=1.2,         #무작위 줌의 범위
    shear_range=0.7,        #층밀리기의 강도
    fill_mode='nearest'     # {"constant", "nearest", "reflect", "wrap"} 디폴트 값은 'nearest'
) 

#테스트 데이터는 rescale만 한다 : 목표가 평가이므로 증폭할 필요가 없음
test_datagen = ImageDataGenerator(
    rescale=1./255
)
xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train/',
    target_size=(100,100),
    batch_size=1000,   #데이터자체에서 미리 batch처리함
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle='True'
    # 폴더에 있는 이미지 데이터를 가져옴(위의 증폭 방식으로)
    #'../_data/brain/train/' x : (160,150,150,1) y:(1600,) /np.unique를 하면 ad:0 이 80, normal:1 이 80개 나옴
    #Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test/',
    target_size=(100,100),
    batch_size=10,   #데이터자체에서 미리 batch처리함
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle='True' #0과 1의 값이 적당히 섞여야 함
    #Found 120 images belonging to 2 classes.
)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten ,Dropout

model = Sequential()
model.add(Conv2D(64, (2,2),input_shape = (100,100,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(36,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(26,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid')) #y값은 0과 1이므로 sigmoid / softmax
# model.add(Dense(2,activation='softmax'))

#3.컴파일, 훈련
model.compile(loss ='binary_crossentropy',optimizer='adam',
              metrics=['acc'])
# model.compile(loss ='sparse_categorical_crossentropy',optimizer='adam',
#               metrics=['acc'])
#4.평가, 검증
hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs =100,
                    validation_data = xy_test,
                    validation_steps=4)
#160개의 데이터 나누기 batch_size =>steps_per_epoch=16
accuracy = hist.history['acc']
val_accuracy = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:',loss[-1]) #가장 마지막의 loss값
print('acc: ',accuracy[-1])
print('val_loss: ',val_loss[-1])


loss = model.evaluate(xy_test)
print('loss : ',loss)



#그래프 그리기
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'

matplotlib.rcParams['axes.unicode_minus'] =False
plt.figure(figsize =(9,6))
plt.plot(hist.history['loss'], c = 'red',
         marker = '.', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue',
         marker = '.',label = 'val_loss')
plt.grid()#격자
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('brain 데이터 손실')
plt.legend(loc = 'upper left')
#plt.legend()

plt.show()



