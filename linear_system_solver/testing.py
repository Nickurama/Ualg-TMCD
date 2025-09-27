import numpy as np


A_a = np.array([
    [ 1, -1,  1],
    [ 2, -2,  2],
    [ 1,  1,  1],
])
B_a = np.array([ 3,  3, -1])

A_b = np.array([
    [ 2, -2,  1],
    [ 2, -2,  2],
    [ 1, -2,  1],
])
B_b = np.array([ 3,  3,  3])

A_c = np.array([
    [ 1, -1, -3,  4],
    [ 1,  1,  1,  2],
    [ 0, -1, -2,  1],
    [ 1,  2,  3,  1]
])
B_c = np.array([ 1, -1,  1, -2])

X = np.linalg.solve(A_b, B_b);
print(X);
