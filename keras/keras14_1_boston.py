#1.train 0.7 이상
#2.R2 :0.8이상 /RMSE사용
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn as sk
print(sk.__version__)

from sklearn.datasets import load_boston

#1. 데이터
dataset= load_boston()
x = dataset.data
y = dataset.target 

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size = 0.7, shuffle = True, random_state= 123
)

print(x)
print(x.shape) #(506,13)
print(y.shape)
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
print(dataset.DESCR)

#2.모델 구성
model = Sequential()
model.add(Dense(1,input_dim=13))




model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))


model.add(Dense(1))
#3.컴파일 , 훈련

model.compile(loss ='mae', optimizer = 'adam',metrics = ['mse'])
model.fit(x_train,y_train,epochs=100, batch_size=1)
#4. 평가, 예측 r2,rmse
loss = model.evaluate(x_test,y_test)
print('loss: ',loss)


y_predict = model.predict(x_test)
print(y_test)
print(y_predict)
from sklearn.metrics import mean_squared_error,r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE: ", RMSE(y_test,y_predict))


r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
