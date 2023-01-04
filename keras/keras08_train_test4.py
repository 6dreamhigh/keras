import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

#1. 데이터
x = np.array([range(10),range(21,31),range(201,211)])
# ctrl+/ 한번에 눌러 주석 처리

x = x.T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])     
y = y.T

# 실습 train_test_split를 이용하여 7:3으로 잘라서 모델 구현 소스 완성
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
   x,y,
    train_size = 0.7, 
    test_size = 0.3,
    shuffle = True,
    #램덤하게 섞는 것->램덤 난수 이용 , False 로 할 경우 list 슬라이싱과 같은 결과
    #default값은 True로 되어 있다.
    random_state = 123
    #실행할 때마다 같은 값이 나오게 한다. 설정 안 할 시에는 할 때마다 다르게 결과가 나온다.  
    
)

#print('x_train:',x_train,'x_test:',x_test,'y_train:',y_train,'y_test:',y_test)
#2.모델 구성
model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train, epochs = 1110,batch_size =1)

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
result = model.predict([[9,30,210]])
print('result : ',result)