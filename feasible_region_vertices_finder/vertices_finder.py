import numpy as np
from itertools import combinations

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

# Example constraints
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
import pandas as pd
df_vertices = pd.DataFrame(vertices, columns=['x1', 'x2'])
import ace_tools as tools; tools.display_dataframe_to_user(name="Feasible Region Vertices", dataframe=df_vertices)

print(df_vertices)
