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





#######################################

import pandas as pd
import numpy as np

df = pd.read_excel('files/manatee.xlsx', header=0)

matriz = df.to_numpy(dtype=float)
n,m = matriz.shape

print("Matriz de dados:\n", matriz)

matriz_media = np.mean(matriz, axis=0)
print("\nMatriz da Media:\n", matriz_media)

matriz_desvio_padrao = np.std(matriz, axis=0)
print("\nMatriz do Desvio padrao:\n", matriz_desvio_padrao)

matriz_standartizada = (matriz - matriz_media) / matriz_desvio_padrao
print("\nMatriz Stndartizada:\n", matriz_standartizada)

matriz_covariancias = (matriz_standartizada.T @ matriz_standartizada) / n
print("\nMatriz das Covariancias:\n", matriz_covariancias)

eigenvalues_mc, eigenvectors_mc = np.linalg.eig(matriz_covariancias)
print("\nMatriz dos Valores proprios da Matriz das Covariancais\n", eigenvalues_mc)
print("\nMatriz dos Vetores Proprios da Matriz das Covariancais\n", eigenvectors_mc)

variancias = (eigenvectors_mc / np.sum(eigenvalues_mc)) * 100
print("\nVariancia:\n", variancias)

num_componentes = 1
V = eigenvectors_mc[:,:num_componentes]

pca = matriz_standartizada @ V
print("\nMatriz transofrmada:\n", pca)
