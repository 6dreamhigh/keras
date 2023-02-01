import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size = 100 #100장 증폭시킨다다

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

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x  /-1의 의미 : 전체데이터
    np.zeros(augument_size),                                                  # y
    batch_size=augument_size,
    shuffle=True
)

print(x_data[0])
print(x_data[0][0].shape)  # (100, 28, 28, 1)
print(x_data[0][1].shape)  # (100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7)) #판사이즈(figsize(가로길이, 세로길이) 단위는 inch이다.)
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()    
