import math
import numpy as np

def length(vector):
    """
    Returns the length of a vector
    """
    return math.sqrt(np.dot(vector, vector))


def compute_angle(a, b):
    """
    Computes the pairwise angles between
    random vectors a and b in R^d
    """
    # if one of the two vectors is a zero vector then return NaN
    if np.all(a == 0) or np.all(b == 0):
        return np.NaN
    return np.arccos(np.dot(a, b)/(length(a)*length(b)))


def angle_data(V: np.ndarray)->np.ndarray:
    """
    Returns an array of pairwise angles between a set of vectors 
    represented as a matrix
    """
    size = V.shape[0]
    result = np.zeros([size*(size-1)//2, 1])
    count = 0
    for i in range(size):
        for j in range(i+1, size):
            if i != j:
                result[count] = compute_angle(V[i,:], V[j, :])
                count +=1
    return result


def build_matrix(V: np.array):
    """
    Builds an array where b[i,j] = 1/(i+j-1)
    """
    n = V.shape[0]
    for i in range(n):
        for j in range(n):
            # indices start with 0
            V[i, j] = 1/(i+1 + j+1 - 1)
    return V