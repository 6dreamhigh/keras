import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM ,SimpleRNN

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

print(x.shape,y.shape) #(7, 3) (7,)
# x = x.reshape(7,3,1) # [[[1],[2],[3]],[[2],[3],[4]],......]
# print(x.shape) #(7,3,1) 3개씩 연산한것을 한개에 상태 전달

#2.모델
model = Sequential()
# model.add(SimpleRNN(units = 64,input_shape=(3,1))) 
model.add(LSTM(units = 64,input_shape = (3,1)))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))
model.summary()


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



'''
(simplernn param) *4 = (lstm parm)
4배의 연산량으로 인해 lstm은 성능은 좋아지지만 속도가 학연히 simplernn보다 더 오래걸리게 된다.

'''