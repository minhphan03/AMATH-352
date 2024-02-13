import numpy as np
import matplotlib.pyplot as plt
import solve

# load data
with np.load(r'mpg_train_test.npz') as data:
    x_train = data['X_train']
    y_train = data['Y_train']
    x_test = data['X_test']
    y_test = data['Y_test']

### Question 1

# find the theta coefficients and compute predictions on test set

def individual_column_qr(degree):
    global x_train, y_train, x_test
    # displacement
    theta_d = solve.compute_coeff_one_feature(x_train, y_train, 1, degree)
    At_d = solve.form_matrix_one_feature(x_test, 1, degree)
    yt_pred_d = At_d.dot(theta_d)

    # horsepower
    theta_h = solve.compute_coeff_one_feature(x_train, y_train, 2, degree)
    At_h = solve.form_matrix_one_feature(x_test, 2, degree)
    yt_pred_h = At_h.dot(theta_h)

    # weight
    theta_w = solve.compute_coeff_one_feature(x_train, y_train, 3, degree)
    At_w = solve.form_matrix_one_feature(x_test, 3, degree)
    yt_pred_w = At_w.dot(theta_w)

    return yt_pred_d, yt_pred_h, yt_pred_w

def individual_column_svd(degree):
    global x_train, y_train, x_test
    # displacement
    theta_d = solve.compute_approximation_matrix(x_train, y_train, 1, degree)
    At_d = solve.form_matrix_one_feature(x_test, 1, degree)
    yt_pred_d = At_d.dot(theta_d)

    # horsepower
    theta_h = solve.compute_approximation_matrix(x_train, y_train, 2, degree)
    At_h = solve.form_matrix_one_feature(x_test, 2, degree)
    yt_pred_h = At_h.dot(theta_h)

    # weight
    theta_w = solve.compute_approximation_matrix(x_train, y_train, 3, degree)
    At_w = solve.form_matrix_one_feature(x_test, 3, degree)
    yt_pred_w = At_w.dot(theta_w)

    return yt_pred_d, yt_pred_h, yt_pred_w

yt_pred_d2, yt_pred_h2, yt_pred_w2 = individual_column_qr(2)

fig, (ax_d, ax_h, ax_w) = plt.subplots(1, 3)
# sort elements
x_graph, y_graph = zip(*sorted(zip(x_test[:,0], yt_pred_d2)))
ax_d.plot(x_graph, y_graph, '-b')
ax_d.plot(x_test[:, 0], y_test, 'or')
ax_d.set_xlabel('displacement')
ax_d.set_ylabel('fuel consumption (mpg)')

x_graph, y_graph = zip(*sorted(zip(x_test[:,1], yt_pred_h2)))
ax_h.plot(x_graph, y_graph, '-b')
ax_h.plot(x_test[:, 1], y_test, 'or')
ax_h.set_xlabel('horsepower')

x_graph, y_graph = zip(*sorted(zip(x_test[:,2], yt_pred_w2)))
ax_w.plot(x_graph, y_graph, '-b')
ax_w.plot(x_test[:, 2], y_test, 'or')
ax_w.set_xlabel('weight')
fig.tight_layout()
plt.show()

### Question 2

## QR implementation
# 4th degree
yt_pred_d4, yt_pred_h4, yt_pred_w4 = individual_column_qr(4)

# 8th degree
yt_pred_d8, yt_pred_h8, yt_pred_w8 = individual_column_qr(8)

# 12th degree
yt_pred_d12, yt_pred_h12, yt_pred_w12 = individual_column_qr(12)

# reporting relative errors
print("Relative errors for displacement models:", solve.relative_error(yt_pred_d2, y_test), solve.relative_error(yt_pred_d4, y_test),
        solve.relative_error(yt_pred_d8, y_test), solve.relative_error(yt_pred_d12, y_test))
print("Relative errors for horsepower models:", solve.relative_error(yt_pred_h2, y_test), solve.relative_error(yt_pred_h4, y_test), 
        solve.relative_error(yt_pred_h8, y_test), solve.relative_error(yt_pred_h12, y_test))
print("Relative errors for weight models:", solve.relative_error(yt_pred_w2, y_test), solve.relative_error(yt_pred_w4, y_test), 
        solve.relative_error(yt_pred_w8, y_test), solve.relative_error(yt_pred_w12, y_test))

'''
## SVD implementation
# 4th degree
yt_pred_d4, yt_pred_h4, yt_pred_w4 = individual_column_svd(4)

# 8th degree
yt_pred_d8, yt_pred_h8, yt_pred_w8 = individual_column_svd(8)

# 12th degree
yt_pred_d12, yt_pred_h12, yt_pred_w12 = individual_column_svd(12)

print(solve.relative_error(yt_pred_d4, y_test), solve.relative_error(yt_pred_d8, y_test), solve.relative_error(yt_pred_d12, y_test))
print(solve.relative_error(yt_pred_h4, y_test), solve.relative_error(yt_pred_h8, y_test), solve.relative_error(yt_pred_h12, y_test))
print(solve.relative_error(yt_pred_w4, y_test), solve.relative_error(yt_pred_w8, y_test), solve.relative_error(yt_pred_w12, y_test))
'''

# plotting
fig, (ax_d, ax_h, ax_w) = plt.subplots(1, 3)

x_graph, y_graph = zip(*sorted(zip(x_test[:,0], yt_pred_d4)))
ax_d.plot(x_graph, y_graph, '-b', label='K=4')
x_graph, y_graph = zip(*sorted(zip(x_test[:,0], yt_pred_d8)))
ax_d.plot(x_graph, y_graph, '-k', label='K=8')
x_graph, y_graph = zip(*sorted(zip(x_test[:,0], yt_pred_d12)))
ax_d.plot(x_graph, y_graph, '-g', label='K=12')
ax_d.plot(x_test[:, 0], y_test, 'or')
ax_d.set_xlabel('displacement')
ax_d.set_ylabel('fuel consumption (mpg)')
ax_d.legend()

x_graph, y_graph = zip(*sorted(zip(x_test[:,1], yt_pred_h4)))
ax_h.plot(x_graph, y_graph, '-b', label='K=4')
x_graph, y_graph = zip(*sorted(zip(x_test[:,1], yt_pred_h8)))
ax_h.plot(x_graph, y_graph, '-k', label='K=8')
x_graph, y_graph = zip(*sorted(zip(x_test[:,1], yt_pred_h12)))
ax_h.plot(x_graph, y_graph, '-g', label='K=12')
ax_h.plot(x_test[:, 1], y_test, 'or')
ax_h.set_xlabel('horsepower')
ax_h.legend()

x_graph, y_graph = zip(*sorted(zip(x_test[:,2], yt_pred_w4)))
ax_w.plot(x_graph, y_graph, '-b', label='K=4')
x_graph, y_graph = zip(*sorted(zip(x_test[:,2], yt_pred_w8)))
ax_w.plot(x_graph, y_graph, '-k', label='K=8')
x_graph, y_graph = zip(*sorted(zip(x_test[:,2], yt_pred_w12)))
ax_w.plot(x_graph, y_graph, '-g', label='K=12')
ax_w.plot(x_test[:, 2], y_test, 'or')
ax_w.set_xlabel('weight')
ax_w.legend()
fig.tight_layout()
plt.show()

### Question 3

xtr_d = x_train[:, 0].reshape([254,1])
xtr_h = x_train[:, 1].reshape([254,1])
xtr_w = x_train[:, 2].reshape([254,1])
xt_d = x_test[:, 0].reshape([138,1])
xt_h = x_test[:, 1].reshape([138,1])
xt_w = x_test[:, 2].reshape([138,1])

# y12
A12 = np.hstack((np.ones([254, 1]), xtr_d, xtr_h, \
    np.multiply(xtr_d, xtr_h), np.square(xtr_d), np.square(xtr_h)))
Q12, R12 = np.linalg.qr(A12)
theta_12 = np.linalg.solve(R12, np.dot(Q12.T, y_train))
At_12 = np.hstack((np.ones([138, 1]), xt_d, xt_h, \
    np.multiply(xt_d, xt_h), np.square(xt_d), np.square(xt_h)))
y_pred_3a = At_12.dot(theta_12)
print("3a relative error: ", solve.relative_error(y_pred_3a, y_test))
print("3a condition number: ", np.linalg.cond(R12))

# y13
A13 = np.hstack((np.ones([254, 1]), xtr_d, xtr_w, \
    np.multiply(xtr_d, xtr_w), np.square(xtr_d), np.square(xtr_w)))
Q13, R13 = np.linalg.qr(A13)
theta_13 = np.linalg.solve(R13, np.dot(Q13.T, y_train))
At_13 = np.hstack((np.ones([138, 1]), xt_d, xt_w, \
    np.multiply(xt_d, xt_w), np.square(xt_d), np.square(xt_w)))
y_pred_3b = At_13.dot(theta_13)
print("3b relative error: ", solve.relative_error(y_pred_3b, y_test))
print("3b condition number: ", np.linalg.cond(R13))

# y23
A23 = np.hstack((np.ones([254, 1]), xtr_h, xtr_w, \
    np.multiply(xtr_h, xtr_w), np.square(xtr_h), np.square(xtr_w)))
Q23, R23 = np.linalg.qr(A23)
theta_23 = np.linalg.solve(R23, np.dot(Q23.T, y_train))
At_23 = np.hstack((np.ones([138, 1]), xt_h, xt_w, \
    np.multiply(xt_h, xt_w), np.square(xt_h), np.square(xt_w)))
y_pred_3c = At_23.dot(theta_23)
print("3c relative error: ", solve.relative_error(y_pred_3c, y_test))
print("3c condition number: ", np.linalg.cond(R23))

# y123
A123 = np.hstack((np.ones([254, 1]), xtr_d, xtr_h, xtr_w, \
    np.square(xtr_d), np.square(xtr_h), np.square(xtr_w)))
Q123, R123 = np.linalg.qr(A123)
theta_123 = np.linalg.solve(R123, np.dot(Q123.T, y_train))
At_123 = np.hstack((np.ones([138, 1]), xt_d, xt_h, xt_w, \
    np.square(xt_d), np.square(xt_h), np.square(xt_w)))
y_pred_3d = At_123.dot(theta_123)
print("3d relative error: ", solve.relative_error(y_pred_3d, y_test))
print("3d condition number: ", np.linalg.cond(R123))