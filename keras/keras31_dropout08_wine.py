from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler  #최대값으로 나누는 함수.


#1.데이터
datasets = load_wine()
x= datasets.data
y = datasets.target

# print(x.shape)
# print(y.shape)

y= to_categorical(y)

#(178, 13)
#(178,)

# print(y)
# print(np.unique(y)) #라벨의 유니크한값을 찾음. [0 1 2] y = 0 , 1 , 2 만존재함.
# print(np.unique(y , return_counts=True))  //#[0 1 2 ] 각각의 갯수출력 (array([0, 1, 2]), array([59, 71, 48], dtype=int64


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
#3.컴파일,훈련
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
"""