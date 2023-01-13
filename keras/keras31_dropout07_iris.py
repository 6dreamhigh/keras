from sklearn.datasets import load_iris #꽃의정보
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler  #최대값으로 나누는 함수.

#1.데이터
datasets = load_iris()
print(datasets.DESCR)    #판다스 .describe()  /  .info()
#input_dim =4
#x=4 y=1
print(datasets.feature_names) #판다스 .columns

x = datasets.data
y = datasets['target']
# print(x)
# print(y)
# print(x.shape)# (150, 4)
# print(y.shape)# (150,)

y = to_categorical(y)   #to_categorical(150,) -> (150,3)




x_train , x_test , y_train , y_test = train_test_split(
    x,y,shuffle=True,  #False의 문제점 y값이 2 하나로 나와 예측값측정이 어려워짐.
    random_state=333,
    test_size=0.2,  #데이터의비율을 비슷하게맞춰줘야함.  한쪾으로 치우친 값을 넣으면 정확도가 떨어짐. 
    stratify=y  # 동일한 비율로 데이터가 넣어짐. [1 0 0 2 1 1 0 2 2 0 2 1 2 1 0] 2:5개 1:5개 0 :5개
                # y의데이터가 분류형 데이터일경우에만 가능. 분류형???
)



scaler = MinMaxScaler()  #데이터 분포가 괜찮으면 MinMaxScaler()
# scaler = StandardScaler()  # 한쪽으로 치우친 데이터 StandardScaler()
scaler.fit(x_train) #x에값에 범위만큼에 가중치생성.(값변환은 안일어남)
x_train=scaler.transform(x_train) #실제 값 변환.
# x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test) #실제 값 변환.  x_train핏한 범위에맞춤.






# print(y_train)
# print(y_test)
#2.모델구성

model = Sequential()
model.add(Dense(50,input_shape=(9,)))
model.add(Dropout(0.5))   # 14 * 50 연산의 50% 훈련시키는과정중엔 적용되지만 평가할땐 적용안되고 전체노드로 적용됨.
model.add(Dense(30))
model.add(Dropout(0.3))
model.add(Dense(60))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(1))
model.summary()

                                           
           
"""                                   
 #3. 컴파일 , 훈련 
# model.compile(loss='categorical_crossentropy', optimizer='adam',  #다중분류 : categorical_crossentropy
#               metrics=['accuracy']
#               )
model.compile(loss='categorical_crossentropy', optimizer='adam',  #다중분류 : categorical_crossentropy
              metrics=['accuracy']
              )
model.fit(x_train ,y_train , epochs=10 , batch_size=1,
          validation_split=0.2,
          verbose=1
          )
#4.평가, 예측
loss,accuracy = model.evaluate(x_test , y_test)
print('loss:',loss)
print('accuracy:' , accuracy)
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)
from sklearn.metrics import accuracy_score
import numpy as np
y_predict=model.predict(x_test)
y_predict = np.argmax(y_predict , axis=1)  # y_predict중 가장 큰값 위치 뽑아내기. 
print('y_pred(예측값):',y_predict)
y_test = np.argmax(y_test, axis=1)# y_test중 가장 큰값 위치 뽑아내기.   
print('y_test(원래값):',y_test)
# acc = accuracy_score(y_test, y_predict)
# print(acc)
# [[1. 0. 0.]  0   원래값.
#  [0. 0. 1.]  2
#  [1. 0. 0.]  0
#  [0. 1. 0.]  1
#  [0. 1. 0.]] 1
# [[9.9765378e-01 2.3461974e-03 1.0845467e-08]   0  예측값.
#  [5.7331777e-06 4.4018906e-03 9.9559230e-01]   2
#  [9.9879801e-01 1.2019606e-03 3.4066683e-09]   0
#  [7.9464291e-05 5.6450292e-02 9.4347024e-01]   2
#  [1.8350199e-04 1.2109222e-01 8.7872428e-01]]  2
"""