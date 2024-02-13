import numpy as np
import matplotlib.pyplot as plt
import solve

# load data
with np.load(r'voting_data_train_test.npz') as data:
    x_train = data['X_train'] 
    y_train = data['Y_train'] 
    x_test = data['X_test'] 
    y_test = data['Y_test']

### Question 1 & 2

N = 16  # number of attributes to take for training

def train_test_mse(N):
    # find the linear function of the divider
    alpha = solve.compute_coeff(x_train, y_train, N)

    # check prediction quality
    A = solve.form_matrix(x_train, N)
    y_pred = np.sign(A.dot(alpha))

    # report training MSE
    MSE_train = solve.compute_mse(y_pred, y_train, A)

    # report test set MSE
    At = solve.form_matrix(x_test, N)
    yt_pred = np.sign(At.dot(alpha))

    MSE_test = solve.compute_mse(yt_pred, y_test, At)

    return MSE_train, MSE_test

mse_train_16, mse_test_16 = train_test_mse(16)
print(mse_train_16, mse_test_16)

### Question 3 

# N = 2
mse_train_2, mse_test_2 = train_test_mse(2)
print(mse_train_2, mse_test_2)

# N = 3
mse_train_3, mse_test_3 = train_test_mse(3)
print(mse_train_3, mse_test_3)

# N = 4
mse_train_4, mse_test_4 = train_test_mse(4)
print(mse_train_4, mse_test_4)