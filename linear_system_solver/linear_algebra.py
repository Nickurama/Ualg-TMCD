import numpy as np
from typing import List, Tuple

# Fazer codigo que resolva exercicio 48 c) (sem usar linalg, usando o
# metodo de gauss jordan, a fazer a matriz condensada), dizer se
# e possivel indeterminada/impossivel/... e o grau de indeterminacao



# ==========================
# constants
# ==========================
PRECISION=0.0001

# ==========================
# helper functions
# ==========================
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

# returns trus if reached end of the matrix
def reached_end(A: np.ndarray, pivot: Tuple[int, int], count_last_col_as_outside: bool = False) -> bool:
    offset = -1
    if not count_last_col_as_outside:
        offset = 0
    return  (pivot[0] >= A.shape[0]) or \
            (pivot[1] >= (A.shape[1] + offset)) or \
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

# returns the value of the pivot
def pivot_value(A: np.ndarray, pivot: Tuple[int, int]) -> float:
    return A[pivot[0], pivot[1]]

# checks the rank of a matrix in reduced row echelon form
def check_rank_rre(A: np.ndarray) -> int:
    rank = 0
    pivot = (0, 0)

    while not reached_end(A, pivot, count_last_col_as_outside=False):
        if is_zero(pivot_value(A, pivot)):
            pivot = move_pivot(pivot, 0, 1)
            continue
        rank += 1
        pivot = move_pivot(pivot, 1, 1)

    return rank

# checks the indexes that are not pivots
def find_undetermined_indexes(pivots: List[Tuple[int, int]], max_len: int) -> List[int]:
    undetermined_indexes: List[int] = []
    sorted_pivots = sorted(pivots, key=lambda x: x[1])
    pivot_i = 0

    for i in range(0, max_len):
        if (pivot_i < len(sorted_pivots)) and (sorted_pivots[pivot_i][1] == i):
            pivot_i += 1
            continue
        undetermined_indexes.append(i)

    return undetermined_indexes

def format_float_if_int(f: float) -> str:
    rounded = np.round(f).astype(int)
    if float_equals(f, float(rounded)):
        return f"{rounded}"
    return f"{f}"

# returns the string solution for the pivot at the given column, for a matrix in rre with
# the pivot values at "+1" (augmented matrix)
def get_sol_str_at_pivot_rre(A: np.ndarray, col: int) -> str:
    row = -1
    for i in range(A.shape[0]):
        if not is_zero(A[i, col]):
            row = i
            break;
    if row == -1:
        raise Exception(f"no value found on col {col}")

    var_str = ""
    first_quotient_sign = ""
    is_first = True
    for i in range(col + 1, A.shape[1] - 1):
        quotient = A[row, i]
        if is_zero(quotient):
            continue
        real_index = i + 1
        quotient = -quotient

        if is_first:
            is_first = False
            if (quotient < 0):
                first_quotient_sign = "-"
            else:
                first_quotient_sign = "+"
        else:
            if (quotient < 0):
                var_str += " - "
            else:
                var_str += " + "

        abs_quotient = abs(quotient)
        if not float_equals(abs_quotient, 1.0):
            var_str += f"{format_float_if_int(abs_quotient)}*x{real_index}"
        else:
            var_str += f"x{real_index}"

    const_str = ""
    constant_term = A[row, A.shape[1] - 1]
    if not is_zero(constant_term):
        const_str += format_float_if_int(constant_term)

    mid_str = ""
    if len(const_str) == 0 and len(var_str) == 0:
        mid_str += "0"
    elif len(const_str) > 0 and len(var_str) == 0:
        mid_str += ""
    elif len(const_str) == 0 and len(var_str) > 0:
        if (first_quotient_sign == "-"):
            mid_str += first_quotient_sign
    else:
        mid_str += f" {first_quotient_sign} "

    return const_str + mid_str + var_str 

def get_sol_str_undetermined_terms(undetermined_indexes: List[int]) -> str:
    if len(undetermined_indexes) == 0:
        return ""

    sol_str = ": "
    is_first = True

    for index in undetermined_indexes:
        real_index = index + 1
        if is_first:
            is_first = False
        else:
            sol_str += ", "

        sol_str += f"x{real_index}"

    sol_str += " in R"

    return sol_str;

def print_solution(A: np.ndarray, rre: np.ndarray, pivots: List[Tuple[int, int]]):
    undetermined_indexes: List[int] = find_undetermined_indexes(pivots, A.shape[1])
    u_i = 0
    is_first = True

    sol_str = "S = { ("
    for i in range (A.shape[1]):
        if is_first:
            is_first = False
        else:
            sol_str += ", "

        if (u_i < len(undetermined_indexes)) and (undetermined_indexes[u_i] == i):
            sol_str += f"x{i + 1}"
            u_i += 1
        else:
            sol_str += get_sol_str_at_pivot_rre(rre, i)
    sol_str += ") "
    sol_str += get_sol_str_undetermined_terms(undetermined_indexes)
    sol_str += " }"
    print(sol_str)







# ==========================
# main functions
# ==========================

# turns an augmented matrix into reduced row echelon form. Returns the reduced row echelon matrix and found pivots
def reduced_row_echelon(A_m: np.ndarray, B_m: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    A = make_augmented_matrix(A_m, B_m)
    pivots: List[Tuple[int, int]] = []
    pivot = (0, 0)

    # row echelon
    while not reached_end(A, pivot, count_last_col_as_outside=True):
        # row swap and column skip
        non_zero_rows = column_find_non_zero_from(A, pivot, invert=False)
        if is_zero(A[pivot[0], pivot[1]]):
            if non_zero_rows == []:
                pivot = move_pivot(pivot, 0, 1)
                continue
            else:
                swap_rows(A, pivot[0], non_zero_rows[0])
        pivots.append((pivot[0], pivot[1])) # pivot found

        # set current pivot to the value "+1"
        multiply_row(A, pivot[0], 1.0 / pivot_value(A, pivot))

        # zero out following rows
        for non_zero_row in non_zero_rows:
            add_scalar_multiple(A, non_zero_row, pivot[0], -A[non_zero_row, pivot[1]])

        # move pivot
        pivot = move_pivot(pivot, 1, 1)

    # reduced row echelon
    for pivot in pivots:
        non_zero_rows = column_find_non_zero_from(A, pivot, invert=True)
        for non_zero_row in non_zero_rows:
            add_scalar_multiple(A, non_zero_row, pivot[0], -A[non_zero_row, pivot[1]])

    return A, pivots

def gauss_jordan(A: np.ndarray, B: np.ndarray, show_info: bool = False) -> None:
    rre, pivots = reduced_row_echelon(A, B)
    format_floats(rre)

    # rank
    rank = len(pivots)
    if show_info:
        print(f"Rank: {rank}")

    # check if impossible
    rank_aug = check_rank_rre(rre)
    if show_info:
        print(f"Augmented rank: {rank_aug}")
    if (rank < rank_aug):
        print("S = Impossible")
        if show_info:
            print(f"\t-> rank(A) < rank(A|B) -> {rank} < {rank_aug} ")
        return;

    # check if possible undetermined (and values + degree of indetermination if info)
    if rank < A.shape[1]:
        print("Possible Undetermined")
        if show_info:
            print(f"\t-> rank(A) < cols(A) -> {rank} < {A.shape[1]} ")
        freedom_degrees = A.shape[1] - rank
        if show_info:
            print(f"Degrees of freedom: {freedom_degrees}")

        print_solution(A, rre, pivots)
        return;


    # possible determinate
    print("Possible Determined")
    print_solution(A, rre, pivots)

    return;

