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
model = load_model(path + 'keras29_1_save_model.h5') #저장된 모델 불러오기 


model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
import numpy as np 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터
dataset = load_boston() #numpy 값을 모여서 사이킷 런 유틸로 묶어놓은 것

x = dataset.data #보스턴 집 값
y = dataset.target



x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    test_size=0.2, shuffle=False

)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
    test_size=0.2, shuffle=True, random_state=333
)



scaler = MinMaxScaler()
#scaler.fit(x_train) 
#x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(26, input_dim=13, activation = 'linear'))
model.add(Dense(40, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(20, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
                                                                                                                                                
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=10, #참을성     
                              restore_best_weights=True, 
                              verbose=1
                              )

model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2)

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

