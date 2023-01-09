from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten
import numpy as np

#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
#print(datasets.feature_names)
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)#(569, 30) (569,)
x_train, x_test,y_train,y_test = train_test_split(
    x,y,
    shuffle = True,
    random_state= 123,
    test_size=0.2
)
#2. 모델
model = Sequential()
model.add(Dense(50, activation = 'linear', input_shape = (30,)))
model.add(Dense(40,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer = 'adam',
              metrics= ['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',mode = 'min',
                              patience=20,restore_best_weights=True,
                              verbose=1) #mode =auto/min/max 보통 min으로 줌

model.fit(x_train, y_train, epochs = 10, batch_size=16,
          callbacks = [earlyStopping], verbose =1)



#4. 평가, 예측
#loss = model.evaluate(x_test,y_test)
#print('loss, accuracy : ',loss)

loss,accuracy = model.evaluate(x_test,y_test)
print('loss : ',loss)
print('accuracy : ',accuracy)
#여러줄 주석 처리 ctrl +/
y_predict1 =model.predict(x_test).flatten()
y_predict = np.where(y_predict1 >0.5, 1,0)
print(y_predict[:10])
print(y_test[:10])
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ',accuracy_score)