#Feature의 개수와 Timestep을 2개로 조정한 경우
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import datetime

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

x_train = x_train.reshape(72,2,2,1) #TIMESTEP =2 /feature = 2
x_test = x_test.reshape(24,2,2,1)
x_predict = x_predict.reshape(7,2,2,1)

print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)


#2. 모델구성
model = Sequential()
model.add(Conv2D(512, (2,2), input_shape=(2, 2, 1)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))                                                                                                
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', 
                              mode='auto', 
                              patience=10,
                              restore_best_weights=True, 
                              verbose=1
                              )


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf

# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'min',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k34_3_cifer100' + date + '_'+ filename)






model.fit(x_train,y_train,epochs=200, batch_size=2)

#4.평가,예측
loss=model.evaluate(x_test,y_test)
print('loss : ', loss)
result = model.predict(x_predict)
print('결과 : ', result)
'''
loss :  7.75969874666771e-06
결과 :  [[100.00454 ]
 [101.00459 ]
 [102.00465 ]
 [103.004684]
 [104.00475 ]
 [105.0048  ]
 [106.00487 ]]


'''