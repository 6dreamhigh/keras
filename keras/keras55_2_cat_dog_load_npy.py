import numpy as np


from tensorflow.keras.preprocessing.image import ImageDataGenerator
#1. 데이터
x_train = np.load('C:/dog vs cat/x_train.npy')
y_train = np.load('C:/dog vs cat/y_train.npy')
x_test = np.load('C:/dog vs cat/x_test.npy')
y_test = np.load('C:/dog vs cat/y_test.npy')

train_datagen = ImageDataGenerator(
    rescale=1./255,                     # 0~255 -> 0~1 사이로 변환 스케일링
    # horizontal_flip=True,               # 수평 반전
    # vertical_flip=True,                 # 수직 반전 
    # width_shift_range=0.1,              # 좌우 이동  0.1만큼 이동
    # height_shift_range=0.1,             # 상하 이동  0.1만큼 이동
    # rotation_range=5,                   # 회전       5도까지 회전 최대 회전 각은 180도
    # zoom_range=1.2,                     # 확대       원래 사이즈의 1.2배까지
    # shear_range=0.7,                    # 기울임     0.7만큼 기울임
    # fill_mode='nearest'                 # 빈자리를 채워줌  nearest: 가장 가까운 값으로 채움 
)

print(x_train.shape, x_test.shape) #(160, 200, 200, 1) (120, 200, 200, 1)
print(y_train.shape, y_test.shape) #(160,) (120,)
dog = train_datagen.flow_from_directory(
    'C:/dog vs cat/dog.PNG/',             # 폴더 경로 지정
    target_size=(200, 200),             # 이미지 사이즈 지정               
    class_mode=None,              # 수치형으로 변환
    # class_mode='categorical',           # one hot encoding
    color_mode='rgb',             # 흑백으로 변환
    shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
    # Found 160 images belonging to 2 classes.
)
cat = train_datagen.flow_from_directory(
    'C:/dog vs cat/cat.PNG/',             # 폴더 경로 지정
    target_size=(200, 200),             # 이미지 사이즈 지정               
    class_mode=None,              # 수치형으로 변환
    # class_mode='categorical',           # one hot encoding
    color_mode='rgb',             # 흑백으로 변환
    shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
    # Found 160 images belonging to 2 classes.
)
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
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
hist = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2)


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



loss2 = model.predict(cat)
loss3 = model.predict(dog)


print(loss2)
print(loss3)
