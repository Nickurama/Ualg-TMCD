import numpy as np

A_i = 3
A_j = 4
B_i = 3
B_j = 4

def build_A(A : np.ndarray) -> np.ndarray:
    rows = A.shape[0]
    columns = A.shape[1]
    for i in range(rows):
        for j in range(columns):
            a_i = i + 1
            a_j = j + 1
            if (i == j):
                A[i, j] = 1
            elif ((a_i + a_j) % 2 == 0):
                A[i, j] = 2
            else:
                A[i, j] = a_i + a_j
    return A

def build_B(B: np.ndarray) -> np.ndarray:
    rows = B.shape[0]
    columns = B.shape[1]
    for i in range(rows):
        for j in range(columns):
            b_i = i + 1
            b_j = j + 1
            B[i, j] = b_i * b_i - b_j * b_j
    return B

def separator() -> None:
    print()
    print("-----------------")
    print()

A = np.empty([A_i, A_j], dtype=float);
B = np.empty([B_i, B_j], dtype=float);

A = build_A(A)
B = build_B(B)

print("Matrix A:\n")
print(A)

separator()

print("Matrix B:\n")
print(B)

separator()

print("A - 2B =\n")
try:
    print(A - 2*B)
except:
    print(f"Cannot subtract matrices, {A.shape[0]}x{A.shape[1]} != {(2*B).shape[0]}x{(2*B).shape[1]}")

separator()

print("AB =\n")
try:
    print(A@B)
except:
    print(f"Cannot multiply matrices, columns of A: {A.shape[1]} != rows of B: {B.shape[0]}")

separator()

print("AB^T =\n")
try:
    print(A@B.transpose())
except:
    print(f"Cannot multiply matrices, columns of A: {A.shape[1]} != rows of B^T: {(B.transpose()).shape[0]}")
