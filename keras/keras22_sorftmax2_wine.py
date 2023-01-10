import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf

#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)
print(y)
print(np.unique(y,return_counts=True))
# y에 대한 0인지 1인지 2인지 찾는다. 
# array([0, 1, 2]), array([59, 71, 48] y는 0과 1과 2로 구분되어 있다.
#y를 원핫처리한 후 output은 3으로 바꾸어줌

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
model.add(Dense(5,activation = 'relu', input_shape = (13,)))
model.add(Dense(24, activation = 'sigmoid'))
model.add(Dense(43, activation = 'relu'))
model.add(Dense(33, activation = 'relu'))
model.add(Dense(22, activation = 'linear'))
model.add(Dense(3, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',
              metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 200, batch_size=1,
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



