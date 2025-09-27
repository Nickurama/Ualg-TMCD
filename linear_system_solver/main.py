import numpy as np
from typing import List, Tuple

# Fazer codigo que resolva exercicio 48 c) (sem usar linalg, usando o
# metodo de gauss jordan, a fazer a matriz condensada), dizer se
# e possivel indeterminada/impossivel/... e o grau de indeterminacao

PRECISION=0.0001

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
def multiply_row(A: np.ndarray, row: int, scalar: float) -> None:
    for i in range(A.shape[1]):
        A[row, i] = A[row, i] * scalar

# 0 indexed
def add_scalar_multiple(A: np.ndarray, row_affected: int, row_added: int, scalar: float) -> None:
    for i in range(A.shape[1]):
        A[row_affected, i] = A[row_affected, i] + A[row_added, i] * scalar

# check if float a is equal to float b within precision
def float_equals(a: float, b: float) -> bool:
    return abs(a - b) < PRECISION

# checks if a float is equal to 0 within precision
def is_zero(a: float) -> bool:
    return float_equals(a, 0.0)

# formats floats so that there are no negative zero values
def format_floats(A: np.ndarray) -> None:
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = round(A[i, j], 2)
            if A[i, j] == -0.0:
                A[i, j] = 0.0

# moves pivot by i_increment and j_increment
def move_pivot(pivot: Tuple[int, int], i_increment: int, j_increment: int) -> Tuple[int, int]:
    return (pivot[0] + i_increment, pivot[1] + j_increment)

# returns trus if reached end of the augmented matrix (constants count as end of matrix)
def reached_end_augmented(A: np.ndarray, pivot: Tuple[int, int]) -> bool:
    return  (pivot[0] >= A.shape[0]) or \
            (pivot[1] >= (A.shape[1] - 1)) or \
            (pivot[0] < 0) or \
            (pivot[1] < 0)

# returns a list with the indexes of all the rows from the pivot onwards that have a non-zero number (or before the pivot if invert = True)
def column_find_non_zero_from(A: np.ndarray, pivot: Tuple[int, int], invert: bool = False) -> List[int]:
    result: List[int] = []
    start = 0 if invert else pivot[0] + 1
    end = pivot[0] if invert else A.shape[0]

    for i in range(start, end):
        if not is_zero(A[i, pivot[1]]):
            result.append(i)

    return result

def pivot_value(A: np.ndarray, pivot: Tuple[int, int]) -> float:
    return A[pivot[0], pivot[1]]





# turns an augmented matrix into row echelon form. Returns a list of the pivots found
def row_echelon(A: np.ndarray) -> List[Tuple[int, int]]:
    pivots: List[Tuple[int, int]] = []
    pivot = (0, 0)

    while not reached_end_augmented(A, pivot):
        # row swap and column skip
        non_zero_rows = column_find_non_zero_from(A, pivot, invert=False)
        if is_zero(A[pivot[0], pivot[1]]):
            if non_zero_rows == []:
                # entire column is zeroes below, move right
                pivot = move_pivot(pivot, 0, 1)
                continue
            else:
                swap_rows(A, pivot[0], non_zero_rows[0])
        pivots.append((pivot[0], pivot[1])) # pivot found

        # set current pivot to +1
        multiply_row(A, pivot[0], 1.0 / pivot_value(A, pivot))

        # zero out following rows
        for non_zero_row in non_zero_rows:
            add_scalar_multiple(A, non_zero_row, pivot[0], -A[non_zero_row, pivot[1]])

        # move pivot
        pivot = move_pivot(pivot, 1, 1)

    return pivots

# turns an augmented matrix that is ALREADY in row echelon into it's reduced row echelon form.
# assumes pivots are set to +1
def reduced_row_echelon(A: np.ndarray, pivots: List[Tuple[int, int]]) -> None:
    for pivot in pivots:
        non_zero_rows = column_find_non_zero_from(A, pivot, invert=True)
        # nullify rows above


def gauss_jordan(A_original: np.ndarray):
    A = np.copy(A_original)
    pivots = row_echelon(A)
    reduced_row_echelon(A, pivots)
    format_floats(A)


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

A_d = np.array([
    [ 3, -2,  5,  1],
    [ 1,  1, -3,  2],
    [ 6,  1, -4,  3],
])
B_d = np.array([ 1,  2,  7])

aug_matrix = make_augmented_matrix(A_d, B_d)
print(aug_matrix)
print()

row_echelon(aug_matrix)
format_floats(aug_matrix)
print(aug_matrix)
print()


# d) solution
# swap_rows(aug, 0, 1)
#
# add_scalar_multiple(aug, 1, 0, -3)
# add_scalar_multiple(aug, 2, 0, -6)
#
# add_scalar_multiple(aug, 2, 1, -1)
#
# multiply_row(aug, 1, -(1/5))
# multiply_row(aug, 2, -(1/4))
#
# add_scalar_multiple(aug, 1, 2, -1)
# add_scalar_multiple(aug, 0, 2, -2)
#
# add_scalar_multiple(aug, 0, 1, -1)
#
# format_doubles(aug)
