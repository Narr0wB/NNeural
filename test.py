import tensorflow as tf
import numpy as np
import math

def eddu():
    g = tf.Variable([[[1,2,4], [3,3,3], [9,9,9]], [[1,2,2], [2,2,1], [87,8,8]]])
    print(g[0])
    return g[0]
#tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



tmp = eddu()
print(tmp)

tmp[1] = [10,10,10]


with tf.device("/GPU:0"):
    a = tf.Variable(np.ones([1, 5]))
    b = tf.add(a, a)
    b = tf.matmul(a, tf.transpose(a))
    print(b)

with tf.device("/CPU:0"):
    print(b[0], b[1])



