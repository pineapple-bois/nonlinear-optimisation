import numpy as np

# Forward substitution function to solve Lc = b
def forward_substitution(L, b):
    """
    Solve the system Lc = b using forward substitution.

    Parameters:
    L (numpy.ndarray): A lower triangular matrix.
    b (numpy.ndarray): The right-hand side vector.

    Returns:
    c (numpy.ndarray): Solution vector.
    """
    n = len(b)
    c = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        if np.isclose(L[i, i], 0):
            raise ValueError(f"Zero or near-zero diagonal element encountered at "
                             f"index {i} during forward substitution.")
        sum_L_c = sum(L[i, k] * c[k] for k in range(i))
        c[i] = (b[i] - sum_L_c) / L[i, i]
    return c

# Backward substitution function to solve Ux = c
def backward_substitution(U, c):
    """
    Solve the system Ux = c using backward substitution.

    Parameters:
    U (numpy.ndarray): An upper triangular matrix.
    c (numpy.ndarray): The right-hand side vector.

    Returns:
    x (numpy.ndarray): Solution vector.
    """
    n = len(c)
    x = np.zeros_like(c, dtype=np.float64)
    for i in range(n-1, -1, -1):
        if np.isclose(U[i, i], 0):
            raise ValueError(f"Zero or near-zero diagonal element encountered at "
                             f"index {i} during backward substitution.")
        sum_U_x = sum(U[i, k] * x[k] for k in range(i+1, n))
        x[i] = (c[i] - sum_U_x) / U[i, i]
    return x


# LU Decomposition function
def lu_decomposition(A):
    n = A.shape[0]  # number of rows (and columns, since A is n x n)
    L = np.zeros((n, n), dtype=np.float64)  # Initialize L as a zero matrix
    U = np.zeros((n, n), dtype=np.float64)  # Initialize U as a zero matrix

    for j in range(n):
        # Step 3: Set u_1j for i=1,...,j and compute the rest
        for i in range(j+1):
            # Calculate the U matrix
            sum_LU = sum(L[i, k] * U[k, j] for k in range(i))
            U[i, j] = A[i, j] - sum_LU

        # Step 4: Set l_ij for i=j+1,...,n and compute the rest
        for i in range(j, n):
            if U[j, j] == 0:
                raise ValueError(f"Zero pivot encountered at U[{j},{j}]. No LU decomposition possible.")
            # Calculate the L matrix
            sum_LU = sum(L[i, k] * U[k, j] for k in range(j))
            L[i, j] = (A[i, j] - sum_LU) / U[j, j]

    # Step 5: Set the diagonal of L to 1
    for i in range(n):
        L[i, i] = 1.0

    return L, U

def plu_decomposition(A, tol=1e-14):
    m_a, n = np.shape(A)
    assert m_a == n, 'LU decomposition called with a non-square matrix'

    L = np.zeros_like(A, dtype=np.float64)  # Initialize L as a zero matrix
    U = np.zeros_like(A, dtype=np.float64)  # Initialize U as a zero matrix

    p = np.arange(n)

    for j in range(n):
        for i in range(j):
            U[i, j] = A[p[i], j] - L[i, :i] @ U[:i, j]

        for i in range(j, n):
            L[i, j] = A[p[i], j] - L[i, :j] @ U[:j, j]

        # Step 5
        # Find the pivot index m
        m = np.argmax(np.abs(L[:, j]))

        # Step 4: Check for singularity
        if abs(L[m, j]) < tol:
            U[j, j] = 0
            L[j, j] = 1
            break

        # Carry out the row swap
        L[[m, j]] = L[[j, m]]
        p[[m, j]] = p[[j, m]]

        U[j, j] = L[j, j]  # Step 6
        L[:, j] /= U[j, j]  # Step 7

    P = np.identity(n, dtype=int)[:, p]

    return P, L, U