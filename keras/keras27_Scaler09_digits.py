from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_digits
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler  #최대값으로 나누는 함수.


#1.데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# print(x.shape , y.shape)   #(1797, 64) (1797,)
# print(np.unique(y,return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
#                                        # y = 9 

y= to_categorical(y)


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



model = Sequential()
model.add(Dense(5 ,activation='relu',input_shape=(64,)))
model.add(Dense(40,activation='sigmoid'))
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(10,activation='softmax')) 



#3.컴파일,훈련

model.compile(loss='categorical_crossentropy', optimizer='adam',  #다중분류 : categorical_crossentropy
              metrics=['accuracy']
              )

model.fit(x_train ,y_train , epochs=10 , batch_size=1,
          validation_split=0.2,
          verbose=1
          )
