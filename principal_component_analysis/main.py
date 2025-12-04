import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)

filename = ""
if len(sys.argv) > 1:
    filename = sys.argv[1]
    print(f"Reading file: {filename}")
else:
    print("Usage: python main.py <filename>")
    sys.exit(1)

df = pd.read_csv(filename)
columns = df.columns
matrix = df.to_numpy()

with open('output.txt', 'w') as file:
    file.write("--------------------\n")
    file.write("| Data matrix read |\n")
    file.write("--------------------\n\n")
    file.write(f"{matrix}\n");
    file.write("\n\n\n\n\n")

    matrix_mean = np.mean(matrix, axis=0)
    file.write("---------------\n")
    file.write("| Matrix mean |\n")
    file.write("---------------\n\n")
    file.write(f"{matrix_mean}\n")
    file.write("\n\n\n\n\n")

    matrix_std_dev = np.std(matrix, axis=0)
    file.write("-----------------------------\n")
    file.write("| Matrix standard deviation |\n")
    file.write("-----------------------------\n\n")
    file.write(f"{matrix_std_dev}\n")
    file.write("\n\n\n\n\n")

    matrix_standardized = (matrix - matrix_mean) / matrix_std_dev
    file.write("-----------------------\n")
    file.write("| Standardized matrix |\n")
    file.write("-----------------------\n\n")
    file.write(f"{matrix_standardized}\n")
    file.write("\n\n\n\n\n")

    covariance_matrix = (matrix_standardized.T @ matrix_standardized) / matrix.shape[0]
    file.write("---------------------\n")
    file.write("| Covariance matrix |\n")
    file.write("---------------------\n\n")
    file.write(f"{covariance_matrix}\n")
    file.write("\n\n\n\n\n")

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    ordered_indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[ordered_indexes]
    eigenvectors = eigenvectors[:,ordered_indexes]

    file.write("---------------------------------\n")
    file.write("| Covariance matrix eigenvalues |\n")
    file.write("---------------------------------\n\n")
    file.write(f"{eigenvalues}\n")
    file.write("\n\n\n\n\n")

    file.write("----------------------------------\n")
    file.write("| Covariance matrix eigenvectors |\n")
    file.write("----------------------------------\n\n")
    file.write(f"{eigenvectors}\n")
    file.write("\n\n\n\n\n")

    component_variances = (eigenvalues / np.sum(eigenvalues)) * 100
    file.write("-----------------------\n")
    file.write("| Component variances |\n")
    file.write("-----------------------\n\n")
    file.write(f"{component_variances}\n")
    file.write("\n\n\n\n\n")

    num_components = 3
    V = eigenvectors[:,:num_components]

    pca = matrix_standardized @ V
    file.write("--------------\n")
    file.write("| New matrix |\n")
    file.write("--------------\n\n")
    file.write(f"{pca}\n")
    file.write("\n\n\n\n\n")

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    x = pca[:, 0]
    y = pca[:, 1]
    z = pca[:, 2]

    ax.scatter(x, y, z, edgecolors='k', color='purple', s=75)

    # Create a grid for the plane at z=0
    x_range = np.linspace(min(x), max(x), 20)
    y_range = np.linspace(min(y), max(y), 20)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)  # Plane at z=0

    # Plot the plane
    ax.plot_surface(X, Y, Z, alpha=0.3, color='gray', edgecolor='none')

    for xi, yi, zi in zip(x, y, z):
        ax.plot([xi,xi], [yi,yi], [zi,0], 'k--', linewidth=1.0)

    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_zlabel("Principal component 3")
    ax.set_title(" " * 75 + "PCA" + " " * 75)
    ax.grid()
    plt.savefig("figure.png", dpi=150, bbox_inches='tight')
    plt.close()

    for i in range(num_components):
        print(f"PC{i + 1} Correlations ({component_variances[i]:.2f}%):")
        correlations = eigenvectors[:,i]
        for j in range(len(columns)):
            extra_len = 25 - len(f"- {columns[j]}: ")
            print(f"\t- {columns[j]}: {correlations[j]:{extra_len}.2f}")
        print()
