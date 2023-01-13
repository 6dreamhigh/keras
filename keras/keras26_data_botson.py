#평가지표 : R2,RMES
# [실습]
# 1.train 0.7이상
# 2.R2 : 0.8이상 / RMES사용
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,StandardScaler  #최대값으로 나누는 함수.


#1.데이터
dataset = load_boston()
x=dataset.data
y=dataset.target

scaler = MinMaxScaler()  #데이터 분포가 괜찮으면 MinMaxScaler()
# scaler = StandardScaler()  # 한쪽으로 치우친 데이터 StandardScaler()
scaler.fit(x) #x에값에 범위만큼에 가중치생성.(값변환은 안일어남)
x=scaler.transform(x) #실제 값 변환.



print(x)
print(type(x))  # <class 'numpy.ndarray'> sklearn에 numpy로 넣어져있음.

# print('최소값:',np.min(x))  #StandardScaler에선 의미없음.
# print('최댓값:',np.max(x))  #StandardScaler에선 의미없음.


"""
변환전:
loss: [83.63468933105469, 6.241967678070068]
RMSE :  9.145200334235554
r2: -0.06809003628495369
변환후 : 
loss: [39.138153076171875, 4.173168659210205]
RMSE :  6.256049128373342
r2: 0.6274923376649518
"""

x_train,x_test,y_train,y_test = train_test_split(x,y,
     train_size=0.7, shuffle=True)

# print(x)
# print(x.shape)  #(506,13)
# print(y)
# print(y.shape)  #(506,)

# print(dataset.feature_names)   #속성의 이름을 가져옴. 
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

# print(dataset.DESCR)

#2.모델구성
#INPUT_DIM  = 13

model = Sequential()
model.add(Dense(5,input_dim=13 , activation='linear'))
model.add(Dense(30 , activation = 'relu'))
model.add(Dense(60 ,activation = 'linear'))
model.add(Dense(50, activation = 'linear'))
model.add(Dense(70, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))

"""
model.add(Dense(5,input_dim=13))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(1))
r2 : 0.77
model.add(Dense(5,input_dim=13))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))
r2 : 0.73
"""
#3.컴파일훈련
model.compile(loss='mse' , optimizer = 'adam', metrics=['mae']) #loss = 'mae' = mean absolute error(절대값으로바꿔서 엔빵)  loss mse(평균제곱오차)
#loss 는 훈련에 영향을 미침. , metrics['mae'] ->[]의미는 2개이상가능.  accuracy = acc 같음 acc는 줄임말.

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=10 ,  #멈추기시작한자리 w를저장후 최적의 w반환
                              restore_best_weights=True, #restore_best_weights : break한시점에 w를저장
                              verbose=1,
                             ) #accuracy : 높으면좋음
                                                            

model.fit(x_train , y_train, epochs=1000 , batch_size=10,validation_split=0.35, callbacks=[earlyStopping])
#4.평가,예측
loss= model.evaluate(x_test , y_test)
print('loss:',loss)

y_predict = model.predict(x_test)

# print('======================')
# print(y_test)
# print(y_predict)
# print('======================')

def RMSE(y_test , y_predict):  #predict : 예측값 y_test  y나눠진값.
    return np.sqrt(mean_squared_error(y_test , y_predict))   #np.sqrt  : sqrt : mse의 루트를 씌운다.

print('RMSE : ' , RMSE(y_test,y_predict))
r2 = r2_score(y_test,y_predict)
print('r2:',r2)  #r2가 높을수록 좋다