import tensorflow as tf
import numpy as np

print(tf.__version__)

# 1. 데이터 준비
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5,input_dim=1,activation = 'relu'))
model.add(Dense(3))
model.add(Dense(1))
#input_dim =1 , 입력 차원이 1이라는 뜻으로, 입력 노드가 한개라는 뜻
#Dense 레이어, 입력과 출력을 모두 연결해주며 입력과 출력을 각각 연결해주는 가중치를 포함
#relu ->은닉층으로 학습 ->은닉층으로 역전파를 통해 좋은 성능이 나오기 때문에 마지막 층이 아니면 거의 relu를 사용
#sigmoid ->이진 분류 문제
#softmax ->확률 값을 이용해 다양한 클래스를 분류하는 문제



#3. 컴파일, 훈련
model.compile(loss='mae',optimizer= 'adam')
#loss는 손실함수로 얼마나 입력데이터가 출력데이터와 일치하는지 평가해주는 함수를 의미
#mse는 평균제곱오차로 얼마나 예측과 다른지 평가
#optimizer는 손실 함수를 기반으로 네트워크를 어떻게 업데이트될지 결정
model.fit(x,y,epochs=3000)
#4. 평가 ,예측
result = model.predict([6])
print('결과:',result)