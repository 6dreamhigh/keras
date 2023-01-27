#앙상블은 여러개의 모델을 병합한 것
import numpy as np
path = '../_data/stock'
x1_datasets = np.array([range(100),range(301,401)]).transpose()
print(x1_datasets)  #(100,2)  삼성전자 시가,고가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T
print(x2_datasets)  #아모레 시가, 고가 종가


y = np.array(range(2001,2101))  #삼성전자의 하루뒤 종가
print(y.shape) #(100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test,x2_train,x2_test,y_train,y_test = train_test_split(
    x1_datasets,x2_datasets,y,train_size=0.7, random_state=1234
)

print(x1_train.shape,x2_train.shape, y_train.shape)
print(x1_test.shape,x2_test.shape, y_test.shape)

#2.모델 구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense , Input

#2-1 모델1.
input1 = Input(shape=(2,))
dense1 = Dense(11,activation='relu', name = 'ds11')(input1)
dense2 = Dense(12,activation='relu', name = 'ds12')(dense1)
dense3 = Dense(13,activation='relu', name = 'ds13')(dense2)
output1 = Dense(14,activation='relu', name = 'ds14')(dense3)



#2-2 모델2.
input2 = Input(shape=(3,))
dense21 = Dense(11,activation='linear', name = 'ds21')(input2)
dense22 = Dense(12,activation='linear', name = 'ds22')(dense21)
output2 = Dense(13,activation='linear', name = 'ds23')(dense22)

#2-3 모델 병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1,output2], name ='mg1')
merge2 = Dense(12, activation='relu',name ='mg2')(merge1)
merge3 = Dense(13,name = 'mg3')(merge2)
last_output = Dense(1,name='last')(merge3) #1은 y컬럼 의미

model = Model(inputs = [input1,input2] , outputs = last_output)

model.summary()

#3.컴파일, 훈련
model.compile(loss ='mse',optimizer = 'adam')
model.fit([x1_train,x2_train],y_train,epochs=10, batch_size=8)


#4.평가 , 예측
loss = model.evaluate([x1_test,x2_test],y_test)
print('loss : ',loss)

'''
메일제목 : 이름 78,700원

삼성 월요일 시가
아모레 5컬럼/ 삼성 5컬럼 사용
'''
#앙상블 소스 / 가중치 전달