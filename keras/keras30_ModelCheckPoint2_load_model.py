import numpy as np 

from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_save/'

# 1. 데이터
dataset = load_boston() 

x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=333
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = load_model( path + 'MCP/keras30_ModelCheckPoint1.hdf5')

#중간 모델구성, 컴파일, 훈련 코드는 생략, 위의 load_model 사용

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ' , mse)
print('mae : ' , mae)


y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))


r2 = r2_score(y_test, y_predict)


print("===================================")
print("R2 : ", r2)
print("RMSE : " , RMSE(y_test, y_predict))
print("===================================")


'''
[MCP 저장]
R2 :  0.8280174179284039
[MCP 로드]
R2 :  0.8280174179284039
'''