import tensorflow as tf
import numpy as np
import math


t1 = np.array([
    [2],
    [2]
])
t2 = np.array([
    [2, 3],
    [1, 1],
    [1, 1],
])

print(np.matmul(t2, t1))

