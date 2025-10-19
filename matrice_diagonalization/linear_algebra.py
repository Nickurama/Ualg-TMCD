import numpy as np
import math
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


# formats float so that there are no negative zero values
def format_float(n: float) -> float:
    result = round(n, 2)
    if result == -0.0:
        result = 0.0
    return result

# formats floats so that there are no negative zero values
def format_floats(A: np.ndarray) -> None:
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = format_float(A[i, j])
            # A[i, j] = round(A[i, j], 2)
            # if A[i, j] == -0.0:
            #     A[i, j] = 0.0

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

# applies the quadratic formula and returns the zeroes
def quadratic_formula(a: float, b: float, c: float) -> List[float]:
    # list with the zeroes
    zeros: List[float] = []

    # quadratic formula when the +- sign is positive
    zeros.append((-b + math.sqrt((b * b) - 4 * a * c)) / (2 * a))
    # quadratic formula when the +- sign is negative
    zeros.append((-b - math.sqrt((b * b) - 4 * a * c)) / (2 * a))

    # returns the found zeroes
    return zeros

# calculates the eigenvalues for a 3x3 matrix, if and only if the first column consists of a number in the first line
# followed by zeroes in that column. This is because it uses the laplace expansion, and would be unable to generalize
# otherwise (as there could be matrices which would have to be expanded into 3rd degree polynomials, which to be solved
# generally via the cubic formula would need imaginary number operations)
def get_eigenvalues_3_by_3(A: np.ndarray) -> List[float]:
    # list for storing all the eigenvalues
    eigenvalues: List[float] = []
    # the first zero would be the negative values of A[0,0], since in laplace expansion it would turn into
    # (A[0][0] - delta) * 1^(1+1) * (rest of expression) + ...
    # and since the rest of the lines in the column are zero, "+ ..." disappears.
    # therefore, the first zero needs to be A[0][0] (A[0][0] - A[0][0] = 0)
    eigenvalues.append(A[0][0]);

    # the matrix we get by "cutting" column 1 and row 1 (we don't need the others since they would be multiplied by 0)
    # has the shape
    # [ x1 - delta        x2     ]
    # [     x3        x4 - delta ]
    x1 = A[1][1]; # note: the actual matrix won't have the "- delta" part
    x2 = A[1][2];
    x3 = A[2][1];
    x4 = A[2][2]; # note: the actual matrix won't have the "- delta" part
    # of which the determinant can be resolved to:
    # delta^2 - (x1 + x4)delta + (x1x4 - x2x3)
    # which is a polynomial of 2nd degree. Ro find the zeros we can solve it using the quadratic formula by plugging the values of a, b and c as:
    # a = 1
    # b = -(x1 + x4)
    # c = x1x4 - x2x3
    # as such:
    a = 1
    b = -(x1 + x4)
    c = x1 * x4 - x2 * x3
    # and applying the quadratic formular
    zeroes = quadratic_formula(a, b, c)
    # and adding them to the eigenvalues
    eigenvalues.append(zeroes[0])
    eigenvalues.append(zeroes[1])

    return eigenvalues

def get_eigenvector(A: np.ndarray, eigenvalue: float) -> np.ndarray:

    # calculate the eigenvector by solving (A - delta I)x = 0, x and 0 being vectors
    # should B be the value of (A - delta I) and delta taking the place of the current eigenvalue:
    B = A - eigenvalue * np.identity(A.shape[0])
    # solves the equation, 0 can be given by np.zeros()
    sol, pivots = reduced_row_echelon(B, np.zeros(B.shape[0]))
    # remove last column because sol is the augmented matrix
    sol = np.delete(sol, sol.shape[1] - 1, axis=1)

    # U is the eigenspace of the eigenvalue, starting out empty
    U = []
    # the index of the last pivot
    curr_pivot = 0
    # for each column
    for col in range(A.shape[1]):
        # get the current pivot
        pivot = pivots[curr_pivot]

        # checks if the current pivot is on this column
        if pivot[1] == col:
            # if so it should skip this column,
            # however it should only iterate the pivot to the next if there is a next one
            if curr_pivot + 1 < len(pivots):
                curr_pivot += 1
            continue

        # if the code reached here, it means the variable on this column is a free variable
        # since there is no pivot

        # appends a row to the eigenspace
        U.append([])
        # copy the solution's column onto the row in U
        for row in range(A.shape[0]):
            U[len(U) - 1].append(format_float(-sol[row][col]))
        # set the value where the current column is to 1
        # (because as this isn't a pivot, it means that this variable is free)
        U[len(U) - 1][col] = 1

    # each eigenvector is on a row for effiency's sake, so we need to transpose everything
    U = np.transpose(U)

    return U


# gets the eigenspace defined by the given eigenvalues
def get_eigenspace(A: np.ndarray, eigenvalues: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    # initializes the D (eigenvalues) and P (eigenspace) matrices
    D = np.zeros(A.shape)
    P = np.array([])

    # the current column of the eigenvalues
    eigen_curr_column = 0

    # calculate the eigenvector of each eigenvalue
    for eigenvalue in eigenvalues:
        # calculate the algebraic multiplicity
        algebraic_multiplicity = 0
        # for each eigenvalue
        for curr in eigenvalues:
            # if that eigenvalue is the same value as the current eigenvalue
            if eigenvalue == curr:
                # it means that there's more algebraic multiplicity
                algebraic_multiplicity += 1

        # gets the eigenspace for the specific eigenvalue
        U = get_eigenvector(A, eigenvalue)
        # calculates the geometric multiplicity, which is just the dimension of the solution space
        geometric_multiplicity = U.shape[1]

        # if algebraic multiplicity isn't the same as the geometric multiplicity for any eigenvalue,
        # it means the matrix isn't diagonizable, which means it should throw an exception, as it is not possible
        if algebraic_multiplicity != geometric_multiplicity:
            raise Exception(f"algebraic multiplicity != geometric multiplicity ({algebraic_multiplicity} != {geometric_multiplicity})! The matrix is not diagonizable")

        # fills out the diagonal with the eigenvalues (D) with the current eigenvalue.
        # The current eigenvalue should show up as many times as it has algebraic multiplicity
        for i in range(algebraic_multiplicity):
            # sets the current diagonal to the eigenvalue
            D[eigen_curr_column][eigen_curr_column] = eigenvalue
            # goes to the next column
            eigen_curr_column += 1

        # if P is empty, then it should just be the current eigenspace
        if P.size <= 0:
            P = U
        # if P is already filled, the current eigenspace should be appended as columns (to the right)
        else:
            P = np.hstack((P, U))

    # if no exception was thrown, it means that algebraic multiplicity was equal to geometric multiplicity for all eigenvalues
    print("matrix is diagonizable, algebraic multiplicity = geometric multiplicity for all eigenvalues")
    return D, P
