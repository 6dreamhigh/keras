from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
#1.데이터
datasets  = load_iris()
print(datasets.DESCR)#input_dim =4 개, output = 1 파악   판다스 : .decribe() / .info()
print(datasets.feature_names) #판다스 ./columns

# x = tf.keras.layers.Input(shape=[4])
# y = tf.keras.layers.Dense(3,activation ='softmax')(x)
# model = tf.keras.models.Model(x,y)

x = datasets.data
y = datasets['target']
from sklearn.preprocessing import OneHotEncoder
import numpy as np

onehot = OneHotEncoder()
onehot.fit(y.reshape(-1, 1))
y = onehot.transform(y.reshape(-1, 1)).toarray()
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)#원핫인 코딩
# print(y[:10])
# print(y.shape)#(150, 3)
# print(x)
# print(y)
# print(x.shape)#(150, 4)
# print(y.shape)#(150,)

x_train, x_test, y_train,  y_test = train_test_split(
    x,y, shuffle= True, random_state=123,test_size=0.2,
    stratify =y
)
# print(y_train)#항상 데이터의 상태 확인하기
# print(y_test) 

#2.   모델 구성
model = Sequential()
model.add(Dense(50, activation='relu',input_shape = (4,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='linear'))
model.add(Dense(3, activation='softmax'))#y의 class의 개수 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer = 'adam',
              metrics= ['accuracy'])
model.fit(x_train, y_train, epochs = 50, batch_size = 1,
          validation_split=0.2, verbose=1)
#4.평가 예측
loss,accuracy = model.evaluate(x_test, y_test)
print('loss : ',loss)
print('accuracy : ',accuracy)

print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis = 1)
print('y_predict(예측값) : ',y_predict)
y_test = np.argmax(y_test, axis = 1)
print('y_test(원래값) : ',y_test)
acc = accuracy_score(y_test,y_predict)
print('accuracy ; ', acc)
# y_predict :  [1 2 0 0 2 2 0 2 0 2 2 1 0 1 0 1 2 2 2 2 2 0 0 1 0 0 1 1 1 1]
# y_test :     [1 2 0 0 2 2 0 1 0 2 2 1 0 1 0 1 2 2 2 2 2 0 0 1 0 0 1 1 1 1]