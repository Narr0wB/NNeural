import tensorflow as tf
import numpy as np
import math

tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

with tf.device("/GPU:0"):
    a = tf.Variable(np.ones([1, 5]))
    b = tf.add(a, a)
    b = tf.matmul(a, tf.transpose(a))

with tf.device("/CPU:0"):
    print(b[0], b[1])



