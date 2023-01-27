import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1, 101))

#예상 y = 100, 107

timesteps = 5  #x는 4개, y는 1개

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape) #(96, 5)

x=bbb[:, :-1]
y=bbb[:, -1] 
print(x,y)

print(x.shape, y.shape) #(96, 4) (96,)

x = x.reshape(96,4,1)


x_predict = np.array(range(96, 106))

timesteps = 4  

def split_y(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x_predict = split_y(x_predict, timesteps)
print(x_predict)
print(x_predict.shape) #(7, 4)


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(
    x,y,shuffle=True, random_state=1234
)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

x_train = x_train.reshape(72,4,1)
x_test = x_test.reshape(24,4,1)
x_predict = x_predict.reshape(7,4,1)

print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)


#2. 모델구성
model = Sequential()
model.add(LSTM(512, input_shape=(4,1), activation='relu')) 
model.add(Dense(256, activation='relu'))                                                                                                
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train,epochs=10, batch_size=2)

#4.평가,예측
loss=model.evaluate(x_test,y_test)
print('loss : ', loss)
result = model.predict(x_predict)
print('결과 : ', result)
'''
loss :  0.017973771318793297
결과 :  [[100.14147]
 [101.14596]
 [102.15054]
 [103.15524]
 [104.16008]
 [105.16501]
 [106.16994]]
 '''