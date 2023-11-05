import tensorflow as tf
import numpy as np
import math


t1 = np.array([
    [2],
    [2],
    [2]
])
t2 = np.array([[
    [2, 3, 3, 4, 5],
    [1, 1, 3, 4, 5],
    [1, 1, 3, 4, 5],
    ], 
    [
    [2, 3, 3, 4, 5],
    [1, 1, 3, 4, 5],
    [1, 1, 3, 4, 5], 
    ], 
    [
    [2, 3, 3, 4, 5],
    [1, 1, 3, 4, 5],
    [1, 1, 3, 4, 5],
    ]])

print(t2 + t1)

