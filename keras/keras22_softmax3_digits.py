import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']
print(x.shape,y.shape)# (1797, 64) (1797,)= 1797 장에 대해 8*8 사진
print(np.unique(y,return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]
#y의 컬럼은 1개지만 y는 10개의 class이므로 output은 10 ->다중분류
#input은 64개
import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[7])
# plt.show()
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
onehot.fit(y.reshape(-1, 1))
y = onehot.transform(y.reshape(-1, 1)).toarray()

x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle = True,random_state=122, test_size=0.2,
    stratify=y
)
#2.  모델 구성
model = Sequential()
model.add(Dense(70,activation = 'relu', input_shape = (64,)))
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(53, activation = 'relu'))
model.add(Dense(49, activation = 'relu'))
model.add(Dense(22, activation = 'linear'))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',
              metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 100, batch_size=1,
          validation_batch_size=0.2, verbose = 1)
#4.평가, 예측
loss, accuracy  = model.evaluate(x_test,y_test)
print('loss : ',loss)
print('accuracy : ',accuracy)

print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis = 1)
print('y_predict(예측값) : ',y_predict)
y_test = np.argmax(y_test, axis = 1)
print('y_test(원래값) : ',y_test)
acc = accuracy_score(y_test,y_predict)
print('accuracy ; ', acc)