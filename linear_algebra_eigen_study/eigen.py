import numpy as np
import pandas as pd


def gram_schmidt_qr(A):
    '''
    Computes the QR decomposition of the matrix A using Gram-Schmidt orthogonalization process.
    
    Parameters:
    A (numpy.ndarray): The input matrix
    
    Returns:
    tuple: Orthogonal matrix Q and upper triangular matrix R
    '''
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - (R[i, j] * Q[:, i])
        
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R


def power_iteration(A, num_simulations=100):
    '''
    Computes the largest eigenvalue and the corresponding eigenvector of the matrix A using power iteration method.
    
    Parameters:
    A (numpy.ndarray): The input matrix
    num_simulations (int): The number of iterations to perform
    
    Returns:
    tuple: The largest eigenvalue and the corresponding eigenvector
    '''
    n, d = A.shape
    b_k = np.random.rand(d)
    
    for _ in range(num_simulations):
        # Compute the matrix-by-vector product Ab
        Ab = np.dot(A, b_k)
        
        # Compute the norm of the vector Ab
        b_k_norm = np.sqrt(np.sum(Ab**2))
        
        # Re normalize the vector
        b_k = Ab / b_k_norm
    
    # Compute the largest eigenvalue
    lambda_k = np.dot(b_k.T, np.dot(A, b_k))
    
    return lambda_k, b_k


def qr_algorithm(A, num_simulations=100):
    '''
    Computes all eigenvalues and eigenvectors of the matrix A using QR algorithm.
    
    Parameters:
    A (numpy.ndarray): The input matrix
    num_simulations (int): The number of iterations to perform
    
    Returns:
    tuple: A matrix with eigenvalues on the diagonal and a matrix with eigenvectors as columns
    '''
    n, d = A.shape
    
    Q_total = np.eye(d)
    for _ in range(num_simulations):
        # Compute the QR decomposition of A
        Q, R = gram_schmidt_qr(A)
        
        # Compute the product RQ
        A = np.dot(R, Q)
        
        # Accumulate the product of the Q matrices
        Q_total = np.dot(Q_total, Q)
    
    # The eigenvalues are on the diagonal of A
    eigenvalues = np.diag(A)
    
    # The eigenvectors are the columns of Q_total
    eigenvectors = Q_total
    
    return eigenvalues, eigenvectors


if __name__ == '__main__':

    decimals = 3

    # Define the matrix A
    A = np.array([[4, 1], [1, 3]])

    # Use np.linalg.eig to compute the eigenvalues and eigenvectors
    eig_vals_np, eig_vecs_np = np.linalg.eig(A)
    eig_vals_np, eig_vecs_np = np.round(eig_vals_np, decimals), np.round(eig_vecs_np, decimals)

    # Use power_iteration to compute the largest eigenvalue and corresponding eigenvector
    largest_eig_val_power_iter, largest_eig_vec_power_iter = power_iteration(A)
    largest_eig_val_power_iter, largest_eig_vec_power_iter = np.round(largest_eig_val_power_iter, decimals), np.round(largest_eig_vec_power_iter, decimals)

    # Use qr_algorithm to compute all eigenvalues and eigenvectors
    eig_vals_qr, eig_vecs_qr = qr_algorithm(A)
    eig_vals_qr, eig_vecs_qr = np.round(eig_vals_qr, decimals), np.round(eig_vecs_qr, decimals)

    # Print the results
    result = []
    result.append({'type':'np.linalg.eig', 'Eigenvalues':eig_vals_np, 'Eigenvectors':eig_vecs_np})
    result.append({'type':'qr_algorithm', 'Eigenvalues':eig_vals_qr, 'Eigenvectors':eig_vecs_qr})
    result.append({'type':'power_iteration', 'Eigenvalues':largest_eig_val_power_iter, 'Eigenvectors':largest_eig_vec_power_iter})

    result = pd.DataFrame(result)
    print(result)
    print('Note: The "power_iteration" method only computes the largest eigenvalue and its corresponding eigenvector.')