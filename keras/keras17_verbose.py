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
model.add(Dense(5,input_shape=(13,)))#다차원시 사용
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
import time

model.compile(loss='mae', optimizer = 'adam')
start= time.time()
model.fit(x_train,y_train,epochs=50,batch_size = 1,
          validation_split=0.2,
          verbose=3)
end = time.time()
#3. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
print("총 걸린 시간 : ",end-start)
#verbose = 2 총 걸린 시간 :  35.74067425727844
#verbose = 1 총 걸린 시간 :  43.835731983184814
#verbose = 0 총 걸린 시간 :  36.92664957046509
#verbose = 3 총 걸린 시간 :  37.28622913360596

