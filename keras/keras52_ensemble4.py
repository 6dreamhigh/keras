#모델 3개 병합한 앙상블 모델

import numpy as np
path = '../_data/stock'
x1_datasets = np.array([range(100),range(301,401)]).transpose()
print(x1_datasets)  


y = np.array(range(2001,2101))  
print(y.shape) #(100,)
y2 = np.array(range(201,301))
print(y2.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test,\
    y_train,y_test,y2_train, y2_test = train_test_split(
    x1_datasets,\
    y,y2,train_size=0.7, random_state=1234
)

print(x1_train.shape, y_train.shape,y2_train.shape)
print(x1_test.shape, y_test.shape,y2_test.shape)

#2.모델 구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense , Input

#2-1 모델1.
input1 = Input(shape=(2,))
dense1 = Dense(11,activation='relu', name = 'ds11')(input1)
dense2 = Dense(12,activation='relu', name = 'ds12')(dense1)
dense3 = Dense(13,activation='relu', name = 'ds13')(dense2)
output1 = Dense(14,activation='relu', name = 'ds14')(dense3)



#2-2 모델2 분기1.

dense41 = Dense(11,activation='relu', name = 'ds41')(output1)
output2 = Dense(14,activation='relu', name = 'ds42')(dense41)

#2-3 모델3 분기2.

dense51 = Dense(11,activation='relu', name = 'ds51')(output1)
output3 = Dense(14,activation='relu', name = 'ds52')(dense51)



model = Model(inputs = [input1] , outputs = [output2,output3])

model.summary()

#3.컴파일, 훈련
model.compile(loss ='mse',optimizer = 'adam')
model.fit(x1_train,[y_train,y2_train],epochs=10, batch_size=32)




#4.평가 , 예측
loss = model.evaluate(x1_test,[y_test,y2_test])
print('loss : ',loss)