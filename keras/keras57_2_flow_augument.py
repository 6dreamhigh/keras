import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size = 40000 #40000장 증폭시킨다다
randidx = np.random.randint(x_train.shape[0], size=augument_size) #60000개 중에서 40000개의 랜덤한 값을 뽑아낸다
                               #60000
print(randidx) #[28494 19606 21883 ... 14373 17830  4182]
print(len(randidx)) #40000

x_augument = x_train[randidx].copy() #데이터의 원본은 건들지 않고 복사본으로
y_augument = y_train[randidx].copy()
print(x_augument.shape, y_augument.shape) # (40000, 28, 28) (40000,)

x_augument = x_augument.reshape(40000, 28, 28, 1)


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

x_augumented = train_datagen.flow(
    x_augument,  # x
    y_augument,  # y
    # np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x  /-1의 의미 : 전체데이터
    # np.zeros(augument_size),                                                  # y
    batch_size=augument_size,
    shuffle=True
)

print(x_augumented[0][0].shape)  # (40000, 28, 28, 1)
print(x_augumented[0][1].shape)  # (40000,)

x_train = x_train.reshape(60000,28,28,1)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))

print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7)) #판사이즈(figsize(가로길이, 세로길이) 단위는 inch이다.)
# for i in range(49):
#     plt.subplot(7,7,i+1)
#     plt.axis('off')
#     plt.imshow(x_data[0][0][i], cmap='gray')
# plt.show()    