from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler,StandardScaler  #최대값으로 나누는 함수.


#1.데이터

datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x=datasets['data']   # ==   x=datasets.data
y = datasets['target'] #   == y = datasets.target
# print(x.shape , y.shape)  #(569, 30) (569,)
x_train , x_test , y_train , y_test = train_test_split(x,y ,test_size=0.2 , shuffle=True,random_state=123)
#2.모델구성
scaler = MinMaxScaler()  #데이터 분포가 괜찮으면 MinMaxScaler()
# scaler = StandardScaler()  # 한쪽으로 치우친 데이터 StandardScaler()
scaler.fit(x_train) #x에값에 범위만큼에 가중치생성.(값변환은 안일어남)
x_train=scaler.transform(x_train) #실제 값 변환.
# x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test) #실제 값 변환.  x_train핏한 범위에맞춤.


model = Sequential()
model.add(Dense(50,activation='linear',input_shape=(30,)))
model.add(Dense(40,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))  #이진분류할떄 사용하는 activation 함수 sigmoid  0~1사이의값을출력.



#3.컴파일,훈련
model.compile(loss='binary_crossentropy' , optimizer ='adam' ,metrics=['accuracy'])  #binary_crossentropy(2진분류 나오면 loss값)

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=20 ,  #멈추기시작한자리 w를저장후 최적의 w반환
                              restore_best_weights=True, #restore_best_weights : 최소값시점에 멈춤.
                              verbose=1,
                             ) #accuracy : 높으면좋음
model.fit(x_train,y_train , epochs=10000 , batch_size=16 , validation_split=0.2,
          callbacks=[earlyStopping], 
          verbose=1)                 

#4.평가,예측
# loss = model.evaluate(x_test , y_test)
# print('loss,accuracy' , loss )                                                                                              
loss,accuracy =  model.evaluate(x_test , y_test)
print('loss:',loss)
print('accuracy:',accuracy)




# print(y_predict[:10])   # ->정수형으로
# print(y_test[:10])

from sklearn.metrics import r2_score,accuracy_score
from sklearn.metrics import classification_report

preds_1d = model.predict(x_test).flatten() # 차원 펴주기
pred_class = np.where(preds_1d > 0.5, 1 , 0) #0.5보다크면 2, 작으면 1
print(classification_report(y_test, pred_class))
acc = accuracy_score(y_test , pred_class)
print("accuracy score:" , acc)
   
  #loss: [0.6604275107383728, 0.640350878238678]  binary , accuracy
  
  