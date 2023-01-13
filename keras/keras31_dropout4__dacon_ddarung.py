import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path = '../_data/ddarung/'
train_csv = pd.read_csv(path +'train.csv',index_col=0)
test_csv = pd.read_csv(path +'test.csv',index_col=0)
submission = pd.read_csv(path +'submission.csv',index_col=0)

# 결측치 처리 
# 1. 선형 방법을 이용하여 결측치
train_csv = train_csv.interpolate(method='linear', limit_direction='forward')

x = train_csv.drop(['count'], axis=1) # 10개 중 count 컬럼을 제외한 나머지 9개만 inputing
y = train_csv['count']


x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    train_size=0.85, test_size=0.15, shuffle=False
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
    test_size=0.2, shuffle=False
)



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)


# x가 스케일링 했을 때, 나머지 데이터들도 스케일링 진행
test_csv = scaler.transform(test_csv)


#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_shape=(9,)))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(70))
model.add(Dense(1))
model.summary()
import time

model.compile(loss='mae', optimizer='adam')                                 
start = time.time()                                                                            
hist = model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_validation, y_validation), verbose=1) #fit 이 return 한다.
end = time.time()

#3. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

#제출
y_submit = model.predict(test_csv)

# to.csv() 를 사용해서 submission_0105.csv를 완성하시오.
submission['count'] = y_submit
submission.to_csv(path + 'submission_0105.csv')

print('============================================')
print(hist) # <keras.callbacks.History object at 0x00000258175F20A0>
print('============================================')
print(hist.history) # loss, vel-loss 의 변화 형태(딕셔너리 형태|key-value) , value의 형태가 list
print('============================================')
print(hist.history['loss'])
print('============================================')
print(hist.history['val_loss'])
print('============================================')
print('loss : ', loss)
print('============================================')
print('걸린시간 : ', end - start)


plt.figure(figsize=(9,6))
# x 명시 안해도 됨
# hist loss 사용, 색은 red, 선모양은 ., y 선의 이름은 loss
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# hist val_loss 사용, 색은 blue, 선모양은 ., x 선의 이름은 val_loss
plt.plot(hist.history['val_loss'], c='blue',  marker='.' , label='val_loss' )
# 차트 gird 생성
plt.grid() 
# x 축 이름 
plt.xlabel('epochs')
# y 축 이름 
plt.ylabel('loss')
# 차트 제목
plt.title('ddarung loss')
# 그래프 선 이름 표
plt.legend()
#plt.legend(loc='upper right')  그래프 선 이름 표, 위치
# 차트 창 띄우기
plt.show()



