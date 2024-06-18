import numpy as np
from itertools import combinations
import pandas as pd

def find_intersections(A, b):
    """
    Finds all vertices of the feasible region defined by linear constraints.
    
    Parameters:
    A (ndarray): Constraint matrix of shape (m, n)
    b (ndarray): Constraint bounds vector of shape (m,)
    
    Returns:
    list: List of feasible vertices, each represented as an ndarray of shape (n,)
    """
    m, n = A.shape
    vertices = []

    for rows in combinations(range(m), n):
        try:
            B = A[list(rows), :]
            d = b[list(rows)]
            if np.linalg.matrix_rank(B) == n:
                x = np.linalg.solve(B, d)
                if np.all(A @ x <= b) and np.all(x >= 0):  # Feasibility check
                    vertices.append(x)
        except np.linalg.LinAlgError:
            continue

    return vertices

# Example constraints: Ax <= b
A = np.array([
    [1, 2],
    [2, 1],
    [-1, 1],
    [-1, 0],
    [0, -1]
])
b = np.array([5, 6, 1, 0, 0])

vertices = find_intersections(A, b)

# Display the results
df_vertices = pd.DataFrame(vertices, columns=['x1', 'x2'])
print("Feasible Region Vertices:")
print(df_vertices.to_string(index=False))
