import numpy as np

def gs(X):
    """
    Implements the Gram-Schmidt algorithm
    to find a set of orthonormal vectors that makes up
    the space of X
    """
    Y = np.zeros(X.shape)
    for i in range(X.shape[0]):
        q = X[i, :]
        for j in range(i):
            # apply the Gram-Schmidt function but remove the denominator
            # because inner product of normalized vector and itself is 1
            q = q - np.multiply(np.dot(Y[j,:], X[i,:]), Y[j, :])

            # if the returned vector is 0, continue (we need a set of N vectors)
        
        # normalize q
        q = q / np.sqrt(np.dot(q, q))

        # add the vector to Y
        Y[i, :] = q
    
    return Y