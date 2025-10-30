import numpy as np
import linear_algebra as la


# ==========================
# constants
# ==========================

A = np.array([
    [  1,  1,  1,  0],
    [ -1,  1,  0,  0],
    [  1,  2,  1,  1],
])

A = A.T

# ==========================
# main
# ==========================

print("A:")
print(A)

print()
print("----------------------")
print()

Q, R = la.qr_decomposition(A)

print("Q: ")
print (Q)
print()

print("R: ")
print (R)
print()
