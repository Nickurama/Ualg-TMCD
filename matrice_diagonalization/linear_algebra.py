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
    # Check if B has more than one dimension (i.e., is not a 1D vector)
    if len(B.shape) > 1:
        # If so, raise an exception since we expect B to be a vector
        raise Exception("B has to be a vector")
    # Check if the number of rows in A matches the length of vector B
    if A.shape[0] != len(B):
        # If not, raise an exception because dimensions are incompatible
        raise Exception("Cannot append a vector with a different number of rows")

    # Create an empty matrix with the same number of rows as A
    # and one extra column to accommodate the constants vector B
    A_B = np.empty([A.shape[0], A.shape[1] + 1], dtype=float);

    # copy A into A_B
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_B[i, j] = A[i, j]

    # copy B into the last column of A_B
    for i in range(len(B)):
        A_B[i, A.shape[1]] = B[i]

    return A_B

# 0 indexed
def swap_rows(A: np.ndarray, row0: int, row1: int) -> None:
    # swaps the row0 with the row1 of the given matrix
    for i in range(A.shape[1]):
        # for each column on the rows, swap the two values
        tmp = A[row0, i]
        A[row0, i] = A[row1, i]
        A[row1, i] = tmp

# 0 indexed
def multiply_row(A: np.ndarray, row: int, scalar: float) -> None:
    # multiplies every value on the row with a scalar
    for i in range(A.shape[1]):
        A[row, i] = A[row, i] * scalar

# 0 indexed
def add_scalar_multiple(A: np.ndarray, row_affected: int, row_added: int, scalar: float) -> None:
    # multiplies row_added by the given scalar and adds the resulting row onto row_affected
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

# moves pivot by i_increment and j_increment
def move_pivot(pivot: Tuple[int, int], i_increment: int, j_increment: int) -> Tuple[int, int]:
    return (pivot[0] + i_increment, pivot[1] + j_increment)

# returns true if reached end of the matrix
def reached_end(A: np.ndarray, pivot: Tuple[int, int], count_last_col_as_outside: bool = False) -> bool:
    offset = -1
    if not count_last_col_as_outside:
        offset = 0
    # checks if the rows and columns go out of the matrix
    return  (pivot[0] >= A.shape[0]) or \
            (pivot[1] >= (A.shape[1] + offset)) or \
            (pivot[0] < 0) or \
            (pivot[1] < 0)

# returns a list with the indexes of all the rows from the pivot onwards that have a non-zero number (or before the pivot if invert = True)
def column_find_non_zero_from(A: np.ndarray, pivot: Tuple[int, int], invert: bool = False) -> List[int]:
    # list of all non_zero rows
    result: List[int] = []
    # goes from 0 -> pivot row if inverted
    # goes from pivot -> end if not inverted
    start = 0 if invert else pivot[0] + 1
    end = pivot[0] if invert else A.shape[0]

    # for each value from start to end
    for i in range(start, end):
        # checks if there is a non-zero value
        if not is_zero(A[i, pivot[1]]):
            # at which it appends the index to the result
            result.append(i)

    return result

# returns the value of the pivot
def pivot_value(A: np.ndarray, pivot: Tuple[int, int]) -> float:
    return A[pivot[0], pivot[1]]

# checks the rank of a matrix in reduced row echelon form
def check_rank_rre(A: np.ndarray) -> int:
    # starts the rank at 0 and the pivot at (0, 0)
    rank = 0
    pivot = (0, 0)

    # while it hasn't gotten out of the matrix
    while not reached_end(A, pivot, count_last_col_as_outside=False):
        # if the current value at the pivot is zero
        if is_zero(pivot_value(A, pivot)):
            # moves pivot to the right by 1 and goes to the next iteration
            pivot = move_pivot(pivot, 0, 1)
            continue
        # if the current value at the pivot is a number
        # increment the rank
        rank += 1
        # and move the pivot to the right and below by 1
        pivot = move_pivot(pivot, 1, 1)

    return rank

# checks the indexes that are not pivots
def find_undetermined_indexes(pivots: List[Tuple[int, int]], max_len: int) -> List[int]:
    # initializes list of indexes that are undetermined
    undetermined_indexes: List[int] = []
    # initializes a list of sorted pivots
    sorted_pivots = sorted(pivots, key=lambda x: x[1])
    # current pivot
    pivot_i = 0

    # for each column until max_len
    for i in range(0, max_len):
        # if the current pivot is valid and the column is the same as the current column
        if (pivot_i < len(sorted_pivots)) and (sorted_pivots[pivot_i][1] == i):
            # go to the next pivot and skip to the next iteration
            pivot_i += 1
            continue
        # if not, the current column is undetermined (there is no pivot)
        undetermined_indexes.append(i)

    return undetermined_indexes

def format_float_if_int(f: float) -> str:
    # rounds the float and casts it to integer (dropping the decimal part)
    rounded = np.round(f).astype(int)
    # if the value is equal to the rounded float
    if float_equals(f, float(rounded)):
        # it means that the value is an int, which the decimal part doesn't matter
        return f"{rounded}"
    # if not, it means the value is a float
    return f"{f}"

# returns the string solution for the pivot at the given column, for a matrix in rre with
# the pivot values at "+1" (augmented matrix)
def get_sol_str_at_pivot_rre(A: np.ndarray, col: int) -> str:
    # initialize row variable to -1 (not found)
    row = -1
    # loop through all rows to find which row contains the pivot in the given column
    for i in range(A.shape[0]):
        # check if the current element is non-zero (using a helper function is_zero)
        if not is_zero(A[i, col]):
            # found the pivot row, store the row index
            row = i
            # exit the loop since we found the pivot
            break;
    # check if no pivot was found in the specified column
    if row == -1:
        # raise exception if no pivot exists in this column
        raise Exception(f"no value found on col {col}")

    # initialize string to build the variable part of the solution
    var_str = ""
    # string to track the sign of the first quotient term
    first_quotient_sign = ""
    # flag to track if we're processing the first non-zero term
    is_first = True
    # loop through columns to the right of the pivot column (excluding the constants column)
    for i in range(col + 1, A.shape[1] - 1):
        # get the coefficient value from the pivot row
        quotient = A[row, i]
        # skip if this coefficient is zero
        if is_zero(quotient):
            continue
        # calculate the actual variable index (it's 1-indexed)
        real_index = i + 1
        # negate the quotient because we're solving for x_col = -quotient*x_i
        quotient = -quotient

        # handle the first non-zero term specially
        if is_first:
            # no longer first term after this one
            is_first = False
             # determine the sign for the first term
            if (quotient < 0):
                first_quotient_sign = "-"
            else:
                first_quotient_sign = "+"
        else:
            # for subsequent terms, add the appropriate operator to the string
            if (quotient < 0):
                var_str += " - "
            else:
                var_str += " + "

        # get absolute value for display
        abs_quotient = abs(quotient)
        # check if the coefficient is not 1 (to avoid writing "1*x")
        if not float_equals(abs_quotient, 1.0):
            # format the coefficient (convert to int if whole number) and add to variable string
            var_str += f"{format_float_if_int(abs_quotient)}*x{real_index}"
        else:
            # if coefficient is 1, just write the variable name
            var_str += f"x{real_index}"

    # initialize string for the constant term
    const_str = ""
    # get the constant term from the last column (augmented part)
    constant_term = A[row, A.shape[1] - 1]
    # if constant term is non-zero, format it
    if not is_zero(constant_term):
        const_str += format_float_if_int(constant_term)

    # initialize string for the middle part (operator between constant and variables)
    mid_str = ""
    # handle different cases for the solution string format:
    # case 1: Both constant and variable parts are empty (solution is zero)
    if len(const_str) == 0 and len(var_str) == 0:
        mid_str += "0"
    # case 2: Only constant term exists, no variables
    elif len(const_str) > 0 and len(var_str) == 0:
        # no operator needed
        mid_str += ""
    # case 3: Only variables exist, no constant term
    elif len(const_str) == 0 and len(var_str) > 0:
        # if the first variable term was negative, include the minus sign
        if (first_quotient_sign == "-"):
            mid_str += first_quotient_sign
    # case 4: Both constant and variable parts exist
    else:
        # add the operator with spaces around it
        mid_str += f" {first_quotient_sign} "

    # combine all parts: constant + middle operator + variables
    return const_str + mid_str + var_str 

def get_sol_str_undetermined_terms(undetermined_indexes: List[int]) -> str:
    # if there are no undetermined variables, returns an empty string
    if len(undetermined_indexes) == 0:
        return ""

    # starts the string with the mathematical symbol ":"
    sol_str = ": "
    # flag for the first term
    is_first = True

    # for each undetermined value
    for index in undetermined_indexes:
        # it's real index is +1 since it's 1 indexed
        real_index = index + 1
        # if it's the first it just sets the flag to false
        if is_first:
            is_first = False
        # if it's not, it should append a comma before the term
        else:
            sol_str += ", "

        # appends the variable's name
        sol_str += f"x{real_index}"

    # terminates showing that they belong to the set of real numbers
    sol_str += " in R"

    return sol_str;

def print_solution(A: np.ndarray, rre: np.ndarray, pivots: List[Tuple[int, int]]):
    # gets the indexes where the variables are undetermined
    undetermined_indexes: List[int] = find_undetermined_indexes(pivots, A.shape[1])
    # initializes undetermined index and the flag showing if it's the first part of the solution
    u_i = 0
    is_first = True

    # start of the solution string
    sol_str = "S = { ("
    # for each column
    for i in range (A.shape[1]):
        # if it's the first part, it just sets the flag to false and does nothing
        if is_first:
            is_first = False
        # if it's not however, it will add a comma before the current part
        else:
            sol_str += ", "

        # if the undetermined value is valid and is on the current column
        if (u_i < len(undetermined_indexes)) and (undetermined_indexes[u_i] == i):
            # display the current undetermined variable
            sol_str += f"x{i + 1}"
            # and increment the undetermined variable
            u_i += 1
        else:
            # else, it inserts the solution of the variable
            sol_str += get_sol_str_at_pivot_rre(rre, i)
    # variable string termination
    sol_str += ") "
    # gets the string for the ending part regarding the undetermined terms
    sol_str += get_sol_str_undetermined_terms(undetermined_indexes)
    # solution termination
    sol_str += " }"
    # prints the solution
    print(sol_str)



# ==========================
# main functions
# ==========================

# turns an augmented matrix into reduced row echelon form. Returns the reduced row echelon matrix and found pivots
def reduced_row_echelon(A_m: np.ndarray, B_m: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    # generates the augmented matrix
    A = make_augmented_matrix(A_m, B_m)
    # initializes the list of pivots
    pivots: List[Tuple[int, int]] = []
    # the current pivot
    pivot = (0, 0)

    # row echelon
    # while it hasn't reached the end of the matrix (the last column, not counting with the augmented column)
    while not reached_end(A, pivot, count_last_col_as_outside=True):
        # row swap and column skip
        # get the rows that are not zero below the pivot
        non_zero_rows = column_find_non_zero_from(A, pivot, invert=False)
        # if the value at the pivot is zero
        if is_zero(A[pivot[0], pivot[1]]):
            # and there is no row that has a value
            if non_zero_rows == []:
                # moves the pivot to the next column, staying in the same row
                pivot = move_pivot(pivot, 0, 1)
                # and skips to the next iteration
                continue
            # if there are is a row with a value
            else:
                # swaps with the first row with a value found
                swap_rows(A, pivot[0], non_zero_rows[0])
        # if the code reached here, it means the pivot is valid and can be added to the list
        pivots.append((pivot[0], pivot[1])) # pivot found

        # set current pivot to the value "+1" to facilitate algebra
        multiply_row(A, pivot[0], 1.0 / pivot_value(A, pivot))

        # zero out following rows on the pivot column using the current row
        for non_zero_row in non_zero_rows:
            add_scalar_multiple(A, non_zero_row, pivot[0], -A[non_zero_row, pivot[1]])

        # move pivot to the next row and column
        pivot = move_pivot(pivot, 1, 1)

    # reduced row echelon
    # for each pivot
    for pivot in pivots:
        # finds the rows with a number above the pivot
        non_zero_rows = column_find_non_zero_from(A, pivot, invert=True)
        # for each one of them
        for non_zero_row in non_zero_rows:
            # zero out that value with the pivot's line
            add_scalar_multiple(A, non_zero_row, pivot[0], -A[non_zero_row, pivot[1]])

    return A, pivots

def gauss_jordan(A: np.ndarray, B: np.ndarray, show_info: bool = False) -> None:
    # get the reduced row echelon and the pivots
    rre, pivots = reduced_row_echelon(A, B)
    # format the floats, so that we don't have weird negative zeros or floating point imprecision
    format_floats(rre)

    # the rank is the number of pivots
    rank = len(pivots)
    # displays the rank if the flag is set
    if show_info:
        print(f"Rank: {rank}")

    # check if impossible:

    # checks the rank of the augmented matrix
    rank_aug = check_rank_rre(rre)
    # shows info if the flag is set
    if show_info:
        print(f"Augmented rank: {rank_aug}")
    # if the rank is lesser than the augmented matrix, it means the system is impossible because
    # "0 = (number on last column of augmented matrix)"
    if (rank < rank_aug):
        print("S = Impossible")
        # shows info if the flag is set
        if show_info:
            print(f"\t-> rank(A) < rank(A|B) -> {rank} < {rank_aug} ")
        return;

    # check if possible undetermined (and values + degree of indetermination if info)
    # if the rank is less than the amount of rows/columns it means the system is possible but undetermined
    if rank < A.shape[1]:
        print("Possible Undetermined")
        # shows info if the flag is set
        if show_info:
            print(f"\t-> rank(A) < cols(A) -> {rank} < {A.shape[1]} ")
        # the degrees of freedom can be given by the difference between number of columns and rank
        freedom_degrees = A.shape[1] - rank
        # shows info if the flag is set
        if show_info:
            print(f"Degrees of freedom: {freedom_degrees}")

        # prints the solution
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
