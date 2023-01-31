#가위 바위 보 모델 만들기
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
ImageDataGenerator(이미지 전처리)
이미지를 학습시킬 때 학습데이터의 양이 적을 경우 훈련데이터를 조금씩 변형시켜서 훈련데이터의 양을 늘리는 방식중에 하나이다.
'''
#이미지 데이터를 증폭시킨다
train_datagen = ImageDataGenerator(
    rescale=1./255,                     # 0~255 -> 0~1 사이로 변환 스케일링
    horizontal_flip=True,               # 수평 반전
    vertical_flip=True,                 # 수직 반전 
    width_shift_range=0.1,              # 좌우 이동  0.1만큼 이동
    height_shift_range=0.1,             # 상하 이동  0.1만큼 이동
    rotation_range=5,                   # 회전       5도까지 회전 최대 회전 각은 180도
    zoom_range=1.2,                     # 확대       원래 사이즈의 1.2배까지
    shear_range=0.7,                    # 기울임     0.7만큼 기울임
    fill_mode='nearest'                 # 빈자리를 채워줌  nearest: 가장 가까운 값으로 채움 
)

test_datagen = ImageDataGenerator(
    rescale=1./255                      # 0~255 -> 0~1 사이로 변환 스케일링 / 평가데이터는 증폭을 하지 않는 원본데이터를 사용한다. 
)

xy_train = train_datagen.flow_from_directory(
    'C:/_data/rps/',             # 폴더 경로 지정
    target_size=(200, 200),             # 이미지 사이즈 지정
    batch_size=126,                       
    # class_mode='binary',                # 수치형으로 변환
    class_mode='categorical',         # 원핫형태의 데이터로 변경            
    color_mode='rgb',             # 흑백으로 변환
    shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.  
    #Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    'C:/_data/rps/',
    target_size=(200,200),
    batch_size=126,
    # class_mode='binary',
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
    #Found 120 images belonging to 2 classes.
)
np.save('C:/_data/rps/x_train.npy', arr=xy_train[0][0])
np.save('C:/_data/rps/y_train.npy', arr=xy_train[0][1])
# np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])

np.save('C:/_data/rps/x_test.npy', arr=xy_test[0][0])
np.save('C:/_data/rps/y_test.npy', arr=xy_test[0][1])

x_train = np.load('C:/_data/rps/x_train.npy')
y_train = np.load('C:/_data/rps/y_train.npy')
x_test = np.load('C:/_data/rps/x_test.npy')
y_test = np.load('C:/_data/rps/y_test.npy')

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(256, (3,3), input_shape=(200,200,3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

hist = model.fit(x_train, y_train, batch_size=16, epochs=100, validation_split=0.2)

#4. 평가, 예측
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


print('loss : ', loss[-1]) #loss[100]
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])
