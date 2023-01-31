#53-3복붙 fit 추가
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#1. 데이터

#이미지 데이터를 증폭시켜 훈련시킴
train_datagen = ImageDataGenerator(
    # rescale=1./255,         #크기 재조절 인수. 디폴트 값은 None
    # horizontal_flip=True,   #인풋을 무작위로 가로로 뒤집습니다.
    # vertical_flip=True,     #인풋을 무작위로 세로로 뒤집습니다.
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,       #무작위 회전의 각도 범위
    # zoom_range=1.2,         #무작위 줌의 범위
    # shear_range=0.7,        #층밀리기의 강도
    # fill_mode='nearest'     # {"constant", "nearest", "reflect", "wrap"} 디폴트 값은 'nearest'
) 

#테스트 데이터는 rescale만 한다 : 목표가 평가이므로 증폭할 필요가 없음
test_datagen = ImageDataGenerator(
    rescale=1./255
)
xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train/',
    target_size=(200,200),
    batch_size=1000,   #데이터자체에서 미리 batch처리함
    class_mode='binary', #수치 /원핫인코딩 해줌
    color_mode='grayscale',
    shuffle='True'
)
print(xy_train[0][1])
print(xy_train[0][0].shape) #(160, 100, 100, 1)
print(xy_train[0][1].shape) #(160, 2)


xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test/',
    target_size=(200,200),
    batch_size=10000,   #데이터자체에서 미리 batch처리함
    class_mode='binary', #수치 원래 0,1구성 원핫할 필요없음
    color_mode='grayscale',
    shuffle='True' #0과 1의 값이 적당히 섞여야 함
    #Found 120 images belonging to 2 classes.
)
#원핫하면 공간을 배로 더 차지하게 됨

np.save('../_data/brain/brain_x_train.npy',arr = xy_train[0][0])
np.save('../_data/brain/brain_y_train.npy',arr = xy_train[0][1])
# np.save('../_data/brain/brain_xy_train.npy',arr = xy_train[0])


np.save('../_data/brain/brain_x_test.npy',arr = xy_test[0][0])
np.save('../_data/brain/brain_y_test.npy',arr = xy_test[0][1])

