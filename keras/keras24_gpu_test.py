import tensorflow as tf
print(tf.__version__)#2.7.4

GPUS = tf.config.experimental.list_physical_devices('GPU')
print(GPUS)

if(GPUS):
    print('gpu 존재')
else:
    print("gpu 안 돈다.")
    
    
    
    
    
    