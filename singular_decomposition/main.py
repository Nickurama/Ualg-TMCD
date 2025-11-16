import numpy as np
import linear_algebra as la


# ==========================
# constants
# ==========================

A = np.array([
    [  0,  1,  1],
    [  1,  0,  1],
    [  1,  1,  2],
])

# A = np.array([
#     [  1,  1],
#     [  0,  1],
#     [ -1,  1],
# ])

# ==========================
# main
# ==========================

print("A:")
print(A)
print()

print("----------------------")
print()

U, sigma, V_t = la.singular_decomposition(A)

print("U:")
print(U)
print()
print("----------------------")
print()

print("sigma:")
print(sigma)
print()
print("----------------------")
print()

print("V_t:")
print(V_t)
print()
print("----------------------")
print()

print("U * sigma * V_t:")
reconstructed = U @ sigma @ V_t
la.format_floats(reconstructed)
print(reconstructed)
print()
print("----------------------")
print()

