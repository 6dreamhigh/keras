import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1.데이터
path = './_data/bike/'
train_csv = pd.read_csv(path +'train.csv',index_col=0)
test_csv = pd.read_csv(path +'test.csv',index_col=0)
submission = pd.read_csv(path +'sampleSubmission.csv',index_col=0)



####결측치 처리 1, 제거####
train_csv = train_csv.dropna() 




train_csv = train_csv.drop(['casual'],axis = 1)#test에 없기 때문에 drop
train_csv = train_csv.drop(['registered'],axis = 1)#test에 없기 때문에 drop
x = train_csv.drop(['count'],axis = 1)
print(x)
y = train_csv['count']#결과로 나와야 하는 값
print(y)
#print(train_csv.info())

#2. 모델구성
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle = True,
    random_state=123
)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

model = Sequential()
model.add(Dense(30, input_dim = 8, activation= 'relu')) #default값 : activation = 'linear'
model.add(Dense(42,activation= 'relu'))
model.add(Dense(52, activation= 'relu'))
model.add(Dense(58, activation= 'sigmoid'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(42, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(1, activation= 'linear'))# 맨 마지막 값은 sigmoid쓰면 0과 1사이 값이므로 쓰면 안됨 /sigmoid는 2진 분류에서만 사용

#3. 컴파일 및 훈련

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',mode = 'min',
                              patience=30,restore_best_weights=True,
                              verbose=1) #mode =auto/min/max 보통 min으로 줌
hist= model.fit(x_train,y_train,epochs=10,batch_size = 10,
          validation_split=0.3,
          verbose=1,callbacks = [earlyStopping])


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
plt.title('Bike Loss')
plt.legend(loc = 'upper left')
#plt.legend()

plt.show()

#제출할 것
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)


#. to_csv()를 사용해서 submission_0105.csv완성
#np.savetxt("submission_0106.csv", y_submit, delimiter=",")
submission['count'] = y_submit
#print(submission)
submission.to_csv(path+'submission_0106.csv')


#loss:  22465.91796875
#RMSE:  149.88635614790627