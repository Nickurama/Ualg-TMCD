import numpy as np
import linear_algebra as la


# ==========================
# constants
# ==========================
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





# ==========================
# main
# ==========================

A = A_a
B = B_a

print("A:")
print(A)
print()
print("B:")
print(B)

print()
print("----------------------")
print()

rre, pivots = la.reduced_row_echelon(A, B)
print("Reduced row echelon:")
print(rre)

print()
print("----------------------")
print()

print("Solution:")
print()
la.gauss_jordan(A, B, show_info=True)
