from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
dataset = load_boston()         # 보스턴 집 값에 대한 데이터
x = dataset.data                # 방 넓이, 방 개수 등 → 독립변수
y = dataset.target              # 집 값 → 종속변수

print(x.shape,y.shape)#(506, 13) (506,)

x_train, x_test,y_train,y_test = train_test_split(
    x,y,
    shuffle = True,
    random_state= 123,
    test_size=0.2
)


#2. 모델구성
model = Sequential()
#model.add(Dense(5,input_dim=13))
model.add(Dense(26, input_dim = 13))
model.add(Dense(32))
model.add(Dense(25))
model.add(Dense(51))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',mode = 'min',
                              patience=5,restore_best_weights=True,
                              verbose=1) #mode =auto/min/max 보통 min으로 줌

#earlystopping은 끊은 시점의 weight가 저장되는 문제점 발생
#최적의 weight를 구하는 방법 : restore_best_weights=True 
hist= model.fit(x_train,y_train,epochs=100,batch_size = 1,
          validation_split=0.2,
          verbose=1,
          callbacks = [earlyStopping])

#3. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
#print("-------------------------------------------")
#print(hist) #<keras.callbacks.History object at 0x0000023735EC2B80>
#print("-------------------------------------------")
print(hist.history)#hist안의 loss의 변화하는 값들을 딕셔너리 형태로 나열
#print("-------------------------------------------")


import matplotlib.pyplot as plt
plt.figure(figsize =(9,6))
plt.plot(hist.history['loss'], c = 'red',
         marker = '.', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue',
         marker = '.',label = 'val_loss')
plt.grid()#격자
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Boston Loss')
plt.legend(loc = 'upper left')
#plt.legend()

plt.show()
#val_loss가 오락가락 높이가 달라질 경우 훈련이 제대로 이루어지고 있지 않다.
#항상 기준은 val_loss

