import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape,y.shape)
print(np.unique(y,return_counts=True))# 각 데이터 항목별로 몇개씩 있는지 알려줌
# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), 
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510]

#pandas getdummy이용할 수도

# y = pd.get_dummies(y)
#np.delete방법도 있음
#힌트 .values    .numpy()
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
model.add(Dense(60,activation = 'relu', input_shape = (54,)))
model.add(Dense(54, activation = 'sigmoid'))
model.add(Dense(43, activation = 'relu'))
model.add(Dense(39, activation = 'relu'))
model.add(Dense(22, activation = 'linear'))
model.add(Dense(7, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',
              metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 30, batch_size=60,
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