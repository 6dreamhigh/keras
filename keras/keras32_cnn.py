from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten #이미지는 평면이므로 2차원, 1차원은 Conv1D



''''
kernel_size = 이미지를 자르는 사이즈. -> (2,2)로자르면 - > (4,4)   
filter 10: (4,4)짜리를 10개의필터로늘린다.
다중분류이므로 dense와 연결되야 되기 때문에 flatten()함수를 통해쫙 펴준다.
45개 , 즉 컬럼이 45개 짜리 데이터라고 할 수 있다. 
쫙 피면 열이 된다.
'''


# model = Sequential()       
#                          #input = (N, 5,5,1) 데이터수/가로/세로/색
# model.add(Conv2D(filters=10,kernel_size=(2,2),
#                  input_shape = (5,5,1)))#원래 이미지의 사이즈    -(N, 4,4,10)
# model.add(Conv2D(5,kernel_size=(2,2))) #- (N,3,3,5)
# model.add(Flatten())#(2,2,5)를 행렬형태로 펴줘야함. 연산량은 없다.     -(N,45)
# model.add(Dense(10)) #(N,10)
# model.add(Dense(units=10))#결과가 나옴     (N,1)
# model.summary()
# #여기서 하이퍼파라미터 튜닝은 다 쓰임-이미지가 어떻게 변환되는지만 다름.
# #너무 conv를 많이 할 시에는 특성이 강화되다가 소멸되버린다. 



model = Sequential()       
                                          
model.add(Conv2D(filters=10,kernel_size=(2,2),#Input shape =(batches_size,rows,cols,channels)
                 input_shape = (5,5,1))) #
model.add(Conv2D(5,kernel_size=(2,2))) 
model.add(Flatten())
model.add(Dense(10))#인풋은 (batch_size, input_dim)
model.add(Dense(units=10))
model.summary()