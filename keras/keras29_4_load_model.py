import numpy as np 

from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


#2. 모델구성(함수형)
path = './_save/'
# path = '../_save/'
# path = 'C:/study/_save/' #절대경로

model = load_model(path + 'keras29_1_save_model.h5')
#keras29_3 에서 만들어진 모델을 가지고 load
#R2 :  0.8083557990742595

#3. 컴파일, 훈련



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


# 변환전
# 변환후
# Scaler 적용 후