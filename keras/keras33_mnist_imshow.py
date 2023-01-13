import numpy as np
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)#(60000, 28, 28) (60000,) 흑백 데이터
#(60000, 28,28, 1) 
print(x_test.shape,y_test.shape)#(60000, 28, 28) (60000,) 흑백 데이터
#(10000, 28, 28) (10000,)
print(x_train[0])#5를 나타내는 그림
print(y_train[0])#5

import matplotlib.pyplot as plt
plt.imshow(x_train[5],'gray')
plt.show()




