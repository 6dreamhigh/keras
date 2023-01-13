import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler  #최대값으로 나누는 함수.

#1.데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv' , index_col=0)   #csv파일 읽어오기 path에있는 train.csv를 가져온다.
test_csv = pd.read_csv(path + 'test.csv' ,index_col=0)   #index_col =0 명시하지않으면  index를 데이터로 인식하게되서 인덱스도 데이터로 인식하게됨.  0번째 컬럼은 인덱스로 명시.
submission = pd.read_csv(path+ 'sampleSubmission.csv' , index_col = 0)  #제출용 파일

#결측치 처리 1.삭제 
# print(train_csv.isnull().sum())
# print(train_csv.shape)
train_csv = train_csv.dropna()
x = train_csv.drop(['casual','registered','count'], axis=1)
# print(x)
y = train_csv['count']
print(y)
x_train, x_test , y_train , y_test = train_test_split(x,y,train_size=0.7, shuffle=True , random_state=0)
print(x_train.shape  , x_test.shape)
print(y_train.shape , y_test.shape)


scaler = MinMaxScaler()  #데이터 분포가 괜찮으면 MinMaxScaler()
# scaler = StandardScaler()  # 한쪽으로 치우친 데이터 StandardScaler()
scaler.fit(x_train) #x에값에 범위만큼에 가중치생성.(값변환은 안일어남)
x_train=scaler.transform(x_train) #실제 값 변환.
# x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test) #실제 값 변환.  x_train핏한 범위에맞춤.







#2.모델구성
model =Sequential()
model.add(Dense(5,input_shape=(8,) , activation='relu'))
model.add(Dense(30,activation='relu'))    
model.add(Dense(45,activation='relu'))
model.add(Dense(55,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(45,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(45,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(45,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='linear'))

#3.컴파일, 훈련 
model.compile(loss='mse' , optimizer = 'adam') 

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=5 ,  #멈추기시작한자리 w를저장후 최적의 w반환
                              restore_best_weights=True, #restore_best_weights : break한시점에 w를저장
                              verbose=1,
                             ) #accuracy : 높으면좋음
                                                                 
hist= model.fit(x_train,y_train ,epochs =150, batch_size=1,validation_split=0.2 , verbose=1 , callbacks=[earlyStopping]) 


#4.평가,예측
loss = model.evaluate(x_test , y_test)  
print('loss:' , loss)
y_predict = model.predict(x_test)

print('================================')
print(hist) #<keras.callbacks.History object at 0x0000024BD1D6B940>
print('================================')
print(hist.history)    #hist.histoey  = loss , val_loss 변화값 (리스트형태)
                       #(key, value) = dictionary자료                    
print('============')    
print(hist.history['loss'])  
print('============')    
print(hist.history['val_loss'])  

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker ='.' , label ='loss')  # epochs 값은 정해져있어서 y만넣어줘도됨
plt.plot(hist.history['val_loss'] , c ='blue' , marker ='.' , label = 'val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('kaggle bike loss')
plt.legend()  #라벨의 이름이 표시됨.
# plt.legend(loc='upper left')   #라벨의 위치지정.
plt.show()

#제출할것
y_submit = model.predict(test_csv)
submission['count'] = y_submit
submission.to_csv(path +'submission_0106.csv')