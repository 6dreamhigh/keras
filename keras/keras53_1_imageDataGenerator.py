import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    batch_size=10,   #데이터자체에서 미리 batch처리함
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle='True'
    # 폴더에 있는 이미지 데이터를 가져옴(위의 증폭 방식으로)
    #'../_data/brain/train/' x : (160,150,150,1) y:(1600,) /np.unique를 하면 ad:0 이 80, normal:1 이 80개 나옴
    #Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test/',
    target_size=(200,200),
    batch_size=10,   #데이터자체에서 미리 batch처리함
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle='True'
    #Found 120 images belonging to 2 classes.
)
print(xy_train[0])
print(xy_train)             #<keras.preprocessing.image.DirectoryIterator object at 0x0000021FDC0ECD90>
print(xy_train[0][0].shape) #train에서 batch_size가 나옴  (10, 100, 100, 1) 10은 batch_size  
# 전체 몇개인지 모를때는 batch_size를 매우 큰 값으로 설정하면 가지고 있는 값으로 돌려  총 train이 몇개인지 알 수 있다.
print(xy_train[0][1])       #[1. 1. 0. 0. 1. 0. 1. 0. 1. 0.]
# print(xy_train[1][1].shape) #(10,)
# print(xy_train[15][1].shape) #(10,)

print(type(xy_train))        #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))     #<class 'tuple'> 튜플은 한번 값 생성 후 바꿀 수 없다.
print(type(xy_train[0][0]))  #<class 'numpy.ndarray'> -->tensorflow에서 numpy로 값을 자동으로 바꿔줌
print(type(xy_train[0][1]))  #<class 'numpy.ndarray'>




