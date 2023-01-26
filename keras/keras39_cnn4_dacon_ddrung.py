import numpy as np 
import pandas as pd
import datetime 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint     
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score



#1. 데이터
path = './_data/ddarung/'                    
                                            #index 컬럼은 0번째
train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # [715 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)     #[1459 rows x 10 columns]
submission = pd.read_csv(path + 'submission.csv', index_col=0)  #[715 rows x 1 columns], 2개중 count 컬럼을 제외한 나머지 1개


# 결측치 처리 
# 1. 선형 방법을 이용하여 결측치
train_csv = train_csv.interpolate(method='linear', limit_direction='forward')

x = train_csv.drop(['count'], axis=1) # 10개 중 count 컬럼을 제외한 나머지 9개만 inputing
y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=123
)


#MinMaxScaler : loss -> [28.201770782470703, 28.201770782470703]
#StandardScaler : loss -> [31.34308433532715, 31.34308433532715]

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# x가 스케일링 했을 때, 나머지 데이터들도 스케일링 진행
test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape) #(1167, 9) (292, 9)


x_train = x_train.reshape(1167, 9, 1, 1)
x_test = x_test.reshape(292, 9, 1, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(128, (2,1), input_shape=(9, 1, 1)))
model.add(Conv2D(64, (2,1)))
model.add(Flatten()) #dropout 훈련 시에만 적용된다.
model.add(Dense(32, activation = 'relu'))

model.add(Dense(14, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))


model.compile(loss='mae', optimizer='adam', metrics=['mae'])                                                  
# 모델을 더 이상 학습을 못할 경우(loss, metric등의 개선이 없을 경우), 학습 도중 미리 학습을 종료시키는 콜백함수                                                                                            
es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'min', 
                   patience = 100, #참을성     
                   #restore_best_weights = False, 
                   verbose = 1)
0


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")


filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf


# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k31_04_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=10, 
          batch_size=8, 
          validation_split=0.25, 
          callbacks=[es, mcp])

model.save(path + 'keras31_droup4_save_model.h5')  #모델 저장 (가중치 포함 안됨)


#3. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

#제출
y_submit = model.predict(test_csv.reshape(715, 9, 1, 1))

# to.csv() 를 사용해서 submission_0105.csv를 완성하시오.
submission['count'] = y_submit
submission.to_csv(path + 'submission_0105.csv')

print('loss : ', loss)



