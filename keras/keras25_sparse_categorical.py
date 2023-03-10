from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR) # pandas: .describe() / .info()
# print(datasets.feature_names) # pandas: .columns

x = datasets.data
y = datasets['target']
# print(x.shape, y.shape) # (150, 4), (150,)
# print(y) # 분류 데이터임을 알 수 있음

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state = 44, stratify=y) # shuffle = False 일 때: 값이 치중될 수 있음, stratify = y: 동일한 비율로 
# print(y_train, "\n", y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape = (4,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(3, activation='softmax')) # 확률의 총합 = 1, 다중 분류에서 사용, 보통 출력 층에서 사용, y 클래스의 개수만큼 지정

#3. 컴파일, 훈련
'''
레이블에 대해서 원-핫 인코딩 과정을 생략하고, 
정수값을 가진 레이블에 대해서 다중 클래스 분류를 수행하고 싶다면
다음과 같이 'sparse_categorical_crossentropy'를 사용

'''
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 50, batch_size=1, validation_split=0.2, verbose = 2)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("loss: ", loss, "accuaracy: ", accuracy)

y_predict = np.argmax(model.predict(x_test), axis = 1)
print('y_predict: ', y_predict)

# y_test = np.argmax(y_test, axis = 1)  # one-hot encoding X
print('y_test: ', y_test)

acc = accuracy_score(y_test, y_predict) 
print('acc: ', acc)