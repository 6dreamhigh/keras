
#R2 0.55 ~0.6이상

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data             
y = dataset.target               

# print(x)
# print(x.shape)                  
# print(y)
# print(y.shape)                 
# print(dataset.feature_names)    
# print(dataset.DESCR)            # Missing Attribute Values: 결측치 - 데이터에 값이 없는 것

#2. 모델구성
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=44
)

model = Sequential()
model.add(Dense(16, input_dim = 8))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(258, activation= 'relu'))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(128, activation ='relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일 및 훈련
model.compile(loss='mae', optimizer='adam', metrics  = ['mse'])
model.fit(x_train, y_train, epochs = 200,
          validation_split = 0.2)


#4. 평가 및 예측
loss = model.evaluate(x_test, y_test) 
print('loss: ', loss)
y_predict = model.predict(x_test)
# print('x_test:\n', x_test)
# print('y_predict:\n', y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

#R2 = 0.55