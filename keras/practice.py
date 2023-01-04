import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#1. 데이터
x= np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test, y_train, y_test = train_test_split(
    x,y,
    train_size = 0.7,
    test_size = 0.3, # 둘 중 하나만 적어도 되고 생략해도 됨.
    shuffle = True, 
    random_state = 123
)

'''
x_train = x[:7]
x_test = x[7:]
y_train = y[:-3]
y_test = y[-3:]
'''
#2. 모델 구성

model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae',optimizer = 'adam')
model.fit(x_train,y_train, epochs=1000,batch_size=1)

#4.평가 ,예측
loss = model.evaluate(x_test,y_test)
print('loss',loss)
'''
result = model.predict([11])
print('result: ',result)
'''
y_predict = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_predict,color ='red')
plt.show()
