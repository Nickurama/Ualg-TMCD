import numpy as np
import linear_algebra as la


# ==========================
# constants
# ==========================

A = np.array([
    [-1,  7, -1],
    [ 0,  1,  0],
    [ 0, 15, -2],
])

# ==========================
# main
# ==========================

print("A:")
print(A)

print()
print("----------------------")
print()

eigenvalues = la.get_eigenvalues_3_by_3(A)
D, P = la.get_eigenspace(A, eigenvalues)

print()
print("----------------------")
print()

print("D: ")
print (D)
print()
print("P: ")
print (P)
print()
