import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) #(10,)
#y가 없기 때문에 정해줘야 한다.
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])#훈련 시킬 y 마지막이 10이므로 10을 포함시키면 안된다.
y = np.array([4,5,6,7,8,9,10])
'''
데이터 형태
x    y
123 |4
234 |5
345 |6
456 |7
567 |8
678 |9
789 |10
'''
print(x.shape,y.shape) #(7, 3) (7,)
# x = x.reshape(7,3,1) # [[[1],[2],[3]],[[2],[3],[4]],......]
# print(x.shape) #(7,3,1) 3개씩 연산한것을 한개에 상태 전달

#2.모델
model = Sequential()
                                            #(N, 3, 1) ->([batch, timesteps, feature])
# model.add(SimpleRNN(units = 256,input_shape=(3,1))) 
model.add(SimpleRNN(units = 256,input_length = 3, input_dim= 1))
# model.add(SimpleRNN(units = 256,input_dim = 1, input_length = 3)) #가독성 떨어짐
'''
SimpleRNN의 심각한 단점
-Timestamp의 수가 많을 경우 예측이 잘 되지 않음
-너무 길 경우 갱신되는 가중치가 소멸되버린다.즉 앞쪽 연산이 뒷쪽 연산에 영향을 미치지 않는다.

SimpleRNN의 상위 버전
-LSTM 


'''
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(50,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))
#컴파일, 훈련
model.compile(loss ='mae',optimizer ='adam')
model.fit(x,y,epochs = 1300)

#평가, 예측
loss = model.evaluate(x,y)
print('loss : ',loss)
#:Model was constructed with shape (None, 3, 1) 
y_pred = np.array([8,9,10]).reshape(1,3,1)
result = model.predict(y_pred)
print('[8,9,10]의 결과 :',result)



