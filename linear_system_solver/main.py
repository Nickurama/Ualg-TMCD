import numpy as np

# Fazer codigo que resolva exercicio 48 c) (sem usar linalg, usando o
# metodo de gauss jordan, a fazer a matriz condensada), dizer se
# e possivel indeterminada/impossivel/... e o grau de indeterminacao

def make_augmented_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if len(B.shape) > 1:
        raise Exception("B has to be a vector")
    if A.shape[0] != len(B):
        raise Exception("Cannot append a vector with a different number of rows")

    A_B = np.empty([A.shape[0], A.shape[1] + 1], dtype=float);

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_B[i, j] = A[i, j]

    for i in range(len(B)):
        A_B[i, A.shape[1]] = B[i]

    return A_B

# 0 indexed
def swap_rows(A: np.ndarray, row0: int, row1: int) -> None:
    for i in range(A.shape[1]):
        tmp = A[row0, i]
        A[row0, i] = A[row1, i]
        A[row1, i] = tmp

# 0 indexed
def multiply_row(A: np.ndarray, row: int, scalar: int) -> None:
    for i in range(A.shape[1]):
        A[row, i] = A[row, i] * scalar

# 0 indexed
def add_scalar_multiple(A: np.ndarray, row_affected: int, row_multiple: int, scalar: int) -> None:
    for i in range(A.shape[1]):
        A[row_affected, i] = A[row_affected, i] + A[row_multiple, i] * scalar


# 0 indexed
# FIXME useless
# def check_zeroes_below(A : np.ndarray, i : int, j : int) -> bool:
#     for k in range(A.shape[0] - i): # FIXME
#         print("owo")

# where A is the augmented matrix
# only works if no row reordering needs to be done
def simplified_gauss_jordan(A: np.ndarray):
    if A.shape[0] != A.shape[1] - 1:
        raise Exception("Not an augmented matrix")

    # start at 0,0
    # if the column is all zeroes, stay at the same row but move to the next column
    # if at current column a number exists, swap that row to the current row (if needed)
    # multiply current row such that the pivot is +1
    # apply the current row to all the others below it that have a non-zero number
    # move to a lower column + row by 1
    # if reached end of rows or end or columns - 1, end

    for i in range(A.shape[0]):
        if A[i, i] == 0:


A = np.array([
    [ 1, -1, -3,  4],
    [ 1,  1,  1,  2],
    [ 0, -1, -2,  1],
    [ 1,  2,  3,  1],
])
B = np.array([ 1, -1,  1, -2])

A_B = make_augmented_matrix(A, B)
print(A_B)

# E = E_6 @ E_5 @ E_4 @ E_3 @ E_2 @ E_1 @ E_0
# X = E @ A
#
# print(X);
# print(np.linalg.solve(A, B));
