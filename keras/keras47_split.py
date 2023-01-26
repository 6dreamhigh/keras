import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM ,Dense

a = np.array(range(1,11))
timesteps = 5

def split_x(dataset,timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : (i+ timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a,timesteps)
print(bbb)
print(bbb.shape) # (6, 5)

x = bbb[:,:-1]
y = bbb[:,:-1]

print(x,y)
print(x.shape,y.shape) #(6, 4) (6, 4)

#실습

#lstm 모델 구성

#2. 모델
model = Sequential()
model.add(LSTM(units = 600,input_shape = (4,1)))
model.add(Dense(512,activation='relu'))
model.add(Dense(308,activation='relu'))
model.add(Dense(264,activation='relu'))
model.add(Dense(132,activation='relu'))
model.add(Dense(56,activation='relu'))
model.add(Dense(36,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(4))
model.summary()


#컴파일, 훈련
model.compile(loss ='mae',optimizer ='adam')
model.fit(x,y,epochs = 1450)

#평가, 예측
loss = model.evaluate(x,y)
print('loss : ',loss)
#:Model was constructed with shape (None, 4, 1) 
x_pred = np.array([7,8,9,10]).reshape(1,4,1)
result = model.predict(x_pred)
print('[7,8,9,10]의 결과 :',result)

