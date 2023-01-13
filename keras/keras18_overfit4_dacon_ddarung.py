import numpy as np
import pandas as pd
from tensorflow .keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1.데이터
path = '../_data/ddarung/'
train_csv = pd.read_csv(path +'train.csv',index_col=0)
test_csv = pd.read_csv(path +'test.csv',index_col=0)
submission = pd.read_csv(path +'submission.csv',index_col=0)

#print(train_csv)
#print(train_csv.shape) #(1459, 10) input_dim = 9 ->count 제외시켜야 함
#print(submission.shape)
#print(train_csv.columns) 
'''
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
      dtype='object')
'''
#print(train_csv.info())
'''
Data columns (total 10 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1459 non-null   int64
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   float64
결측치(데이터가 부족한 경우)가 현재 2개 
-결측치 데이터는 삭제 처리한다. 

'''
####결측치 처리 1, 제거####
print(train_csv.isnull().sum())
train_csv = train_csv.dropna() 
#print(train_csv.shape)
#info와 isnull의 차이점 기억

#print(test_csv.info())
#print(train_csv.describe())

x = train_csv.drop(['count'],axis = 1)#count제거
#print(x)
y = train_csv['count']
#print(y)
#print(y.shape)
#print(x.shape)


#2. 모델구성
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle = True,
    random_state=123
)
#print(x_train.shape,x_test.shape)
#print(y_train.shape,y_test.shape)

model = Sequential()
model.add(Dense(30, input_dim = 9))
model.add(Dense(512,activation = 'relu'))
model.add(Dense(104,activation = 'sigmoid'))
model.add(Dense(512,activation = 'relu'))
model.add(Dense(358,activation = 'relu'))
model.add(Dense(64,activation = 'relu'))
model.add(Dense(42,activation = 'relu'))
model.add(Dense(32,activation= 'relu'))
model.add(Dense(32,activation= 'relu'))
model.add(Dense(1))

#3. 컴파일 및 훈련

model.compile(loss='mse', optimizer='adam')
hist= model.fit(x_train,y_train,epochs=150,batch_size = 1,
          validation_split=0.3,
          verbose=1)


#4. 평가 및 예측
loss = model.evaluate(x_test, y_test) 
print('loss: ', loss)
y_predict = model.predict(x_test)
#print('x_test:\n', x_test)
#print('y_predict:\n', y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE: ", rmse)

import matplotlib.pyplot as plt
plt.figure(figsize =(9,6))
plt.plot(hist.history['loss'], c = 'red',
         marker = '.', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue',
         marker = '.',label = 'val_loss')
plt.grid()#격자
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Ddarung Loss')
plt.legend(loc = 'upper left')
#plt.legend()

plt.show()
#제출할 것
y_submit = model.predict(test_csv)



submission['count'] = y_submit
#print(submission)
submission.to_csv(path+'submission_0105.csv')
