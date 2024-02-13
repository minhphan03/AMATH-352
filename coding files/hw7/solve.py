import numpy as np

def form_matrix(x, N):
    """
    returns the matrix formed by a set of linear equations
    in the form f(x) = a0 + a1x1 + a2x2 + ... + aNxN
    """
    M = x.shape[0]
    return np.append(np.ones([M, 1]), x[:, :N], axis=1)


def compute_coeff(x, y, N):
    """
    computes the coefficients of the line dividing the clusters
    """
    A = form_matrix(x, N)
    Q, R = np.linalg.qr(A)

    alpha = np.linalg.solve(R, np.dot(Q.T, y))
    return alpha


def compute_mse(y_pred, y, A):
    """
    computes the mean square error
    of the predictions of the model and real data
    """
    M = A.shape[0]
    return (1/M)*np.linalg.norm(y_pred - y)**2