import numpy as np

A_0 = np.array([
    [ 2, -2,  1],
    [ 2, -2,  2],
    [ 1, -2,  1],
])
B_0 = np.array([ 3,  3,  3])

A_1 = np.array([
    [ 1, -1,  1],
    [ 2, -2,  2],
    [ 1,  1,  1],
])
B_1 = np.array([ 3,  3, -1])

A_2 = np.array([
    [ 1, -1, -3,  4],
    [ 1,  1,  1,  2],
    [ 0, -1, -2,  1],
    [ 1,  2,  3,  1]
])
B_2 = np.array([ 1, -1,  1, -2])

X_0 = np.linalg.solve(A_0, B_0);
print(X_0);
# X_1 = np.linalg.solve(A_1, B_1);
# print(X_1);
# X_2 = np.linalg.solve(A_2, B_2);
# print(X_2);


# Fazer codigo que resolva exercicio 48 c) (sem usar linalg, usando o
# metodo de gauss jordan, a fazer a matriz condensada)
