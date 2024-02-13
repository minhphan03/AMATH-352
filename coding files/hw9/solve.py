import numpy as np

def form_matrix_one_feature(x, N: int, degree: int):
    """
    returns the matrix formed by a set of linear equations
    in the form f(x) = a0 + a1x1 + a2x2 + ... + aNxN
    """
    M = x.shape[0]
    column = x[:, N-1]
    # 2nd order polynomial
    matrix = np.vander(column, degree+1)
    return np.fliplr(matrix)


def compute_coeff_one_feature(x, y, N: int, degree: int):
    """
    computes the coefficients of the line dividing the clusters
    """
    A = form_matrix_one_feature(x, N, degree)
    Q, R = np.linalg.qr(A)

    alpha = np.linalg.solve(R, np.dot(Q.T, y))

    print("Condition number of feature " + str(N) + " of degree " + str(degree) + " is " + str(np.linalg.cond(R)))
    return alpha


def relative_error(y_pred, y_test):
    """
    computes the relative error
    of the predictions of the model and real data
    """
    relative_error = 1/(np.linalg.norm(y_test)**2)*np.sum(np.square(y_pred - y_test), axis=0)
    return relative_error

# SVD implementation
def compute_approximation_matrix(x, y, N: int, degree: int):
    A = form_matrix_one_feature(x, N, degree)
    r = np.linalg.matrix_rank(A)
    n = A.shape[1]
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Moore-Penrose pseudoinverse
    sigma_inv = np.diag(np.hstack([1/s[:r], np.zeros(n-r)]))
    A_plus = np.dot(U, np.dot(sigma_inv, Vt)).T
    # A_plus = np.linalg.pinv(A)
    # print(np.linalg.cond(A))
    alpha = A_plus.dot(y)
    return alpha
 