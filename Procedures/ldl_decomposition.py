import numpy as np
from Procedures.LU_functions import (
    forward_substitution, backward_substitution)


def ldl_decomposition(A, tol=1e-14, verbose=False):
    """
    Perform the LDL decomposition of a symmetric matrix A, with an optional
    check for whether pivoting is necessary.

    Parameters:
    A (numpy.ndarray): A square, symmetric matrix of shape (n, n).
    tol (float): Tolerance below which we consider a pivot too small.

    Returns:
    L (numpy.ndarray): A lower triangular matrix with ones on the diagonal.
    D (numpy.ndarray): A diagonal matrix such that A = L @ D @ L.T.
    p (numpy.ndarray, optional): Permutation array if pivoting is applied,
                                 otherwise None.

    Raises:
    ValueError: If the input matrix A is not square or symmetric.
    """
    # Check if A is square and symmetric
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError("Matrix A must be symmetric.")

    # Eigenvalue check for positive definiteness
    eigenvalues = np.linalg.eigvalsh(A)
    if np.any(eigenvalues <= 0):
        if verbose:
            print("Matrix is not positive definite. Pivoting will be applied.")
        return ldl_decomposition_pivoted(A, tol)

    n = A.shape[0]  # Number of rows/columns
    L = np.eye(n, dtype=np.float64)  # Initialize L as an identity matrix
    D = np.zeros(n, dtype=np.float64)  # Initialize D as a diagonal array

    for j in range(n):
        # Step 1: Compute the diagonal entry D[j]
        sum_LD = sum(L[j, k]**2 * D[k] for k in range(j))
        D[j] = A[j, j] - sum_LD

        # Check if pivoting is necessary due to negative or too small D[j]
        if abs(D[j]) < tol or D[j] < 0:
            print(f"Pivoting required at step {j}.")
            return ldl_decomposition_pivoted(A, tol)

        # Step 2: Compute the L entries below the diagonal in column j
        for i in range(j + 1, n):
            sum_LD = sum(L[i, k] * L[j, k] * D[k] for k in range(j))
            L[i, j] = (A[i, j] - sum_LD) / D[j]

    # Convert D to a diagonal matrix for reconstruction
    D_matrix = np.diag(D)

    return L, D_matrix, None  # None indicates no pivoting


def ldl_decomposition_pivoted(A, tol=1e-14):
    """
    Perform an LDL decomposition with (simple) pivoting of a symmetric matrix A.
    This code illustrates the general idea of pivoting, but does not implement
    the full Bunch-Kaufman algorithm for indefinite matrices.

    Parameters:
    A (numpy.ndarray): A square, symmetric matrix of shape (n, n).
    tol (float): Tolerance below which we consider a pivot too small.

    Returns:
    P (numpy.ndarray): Permutation matrix describing row/column swaps.
    L (numpy.ndarray): Lower triangular matrix with ones on the diagonal.
    D (numpy.ndarray): Diagonal matrix (as a 1D array or 2D diag) such that
                       P^T A P = L @ D @ L.T.

    Raises:
    ValueError: If the input matrix A is not square or symmetric.
    """
    # Check squareness and symmetry
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix A must be square.")
    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError("Matrix A must be symmetric.")

    # Work on a copy since we'll do row/column swaps
    A = A.astype(np.float64).copy()

    # Initialize L, D, and the permutation matrix P
    L = np.eye(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)
    p = np.arange(n, dtype=int)  # pivot array  # keep track of row/column permutations

    for j in range(n):
        # --- Pivot selection step ---
        # Find index k >= j such that |A[k, k]| is maximal
        pivot_index = j + np.argmax(np.abs(A[j:, j]))
        if pivot_index != j:
            # Swap rows j and pivot_index
            A[[j, pivot_index], :] = A[[pivot_index, j], :]
            # Swap columns j and pivot_index
            A[:, [j, pivot_index]] = A[:, [pivot_index, j]]
            # Record pivot in p
            p[j], p[pivot_index] = p[pivot_index], p[j]

            # Also swap rows in L up to j (because only these are filled so far)
            if j > 0:
                L[[j, pivot_index], :j] = L[[pivot_index, j], :j]

        # Now the pivot is at A[j, j]
        pivot_val = A[j, j]
        if abs(pivot_val) < tol:
            raise ValueError(f"Pivot too small (|{pivot_val}| < {tol}) at index {j}")

        # Compute the diagonal entry
        D[j] = pivot_val

        # Compute the L entries below the diagonal in column j
        for i in range(j + 1, n):
            L[i, j] = A[i, j] / pivot_val

        # Update the submatrix A[j+1:, j+1:]
        for i in range(j + 1, n):
            for k in range(j + 1, n):
                A[i, k] -= L[i, j] * A[j, k]

    # Convert D to a 2D diagonal matrix
    D_matrix = np.diag(D)

    # Permute L
    L_permutation = np.eye(A.shape[0])[p] @ L

    return L_permutation, D_matrix, p


def ldl_solve(L, D, p, b):
    """
    Solve the system A x = b for x, given the LDL^T decomposition of A.

    Parameters
    ----------
    L : np.ndarray
        Permuted lower-triangular factor from LDL^T decomposition.
    D : np.ndarray
        Diagonal matrix from LDL^T decomposition.
    p : np.ndarray or None
        Permutation vector from pivoting. If None, no pivoting was applied.
    b : np.ndarray
        Right-hand side vector.

    Returns
    -------
    x : np.ndarray
        The solution vector.
    """
    if p is not None:
        # Permute the right-hand side according to the permutation vector
        b = b[p]

    # Solve Lc = b using forward substitution
    c = forward_substitution(L, b)

    # Solve Dd = c (diagonal system, element-wise division)
    d = c / np.diag(D)

    # Solve L^T x = d using backward substitution
    x = backward_substitution(L.T, d)

    if p is not None:
        # Undo the permutation for the final solution
        x_undo_permutation = np.zeros_like(x)
        x_undo_permutation[p] = x
        return x_undo_permutation

    return x
