import numpy as np


def approximation(A, U, Vt, s, r):
    # s is a 1D array
    s_low_rank = np.zeros(s.shape)
    s_low_rank[0:r] = s[0:r]

    # singular matrix 
    Sigma_r = np.zeros(A.shape)
    np.fill_diagonal(Sigma_r, s_low_rank)
    return np.dot(U, np.dot(Sigma_r, Vt))


def compute_error(A, U, s, Vt, percent_error):
    r = 2
    while True:
        A_r = approximation(A, U, Vt, s, r)
        error = np.linalg.norm(A - A_r)/np.linalg.norm(A)
        if error <= percent_error:
            break
        else:
            r += 1
    return r, A_r