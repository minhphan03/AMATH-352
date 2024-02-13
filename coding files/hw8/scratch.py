import numpy as np
import matplotlib.pyplot as plt

with np.load(r'faces.npz') as data:
    A = data['A']



mean = np.mean(A, 0)
# plt.imshow(np.reshape(mean, (31, 23)), cmap='gray')
# plt.show()

### Question 2
B = np.subtract(A, mean)

fig, axes = plt.subplots(2, 5)
axes[0,0].imshow(A[0,:].reshape((31, 23)), cmap='gray')
axes[0,1].imshow(A[1,:].reshape((31, 23)), cmap='gray')
axes[0,2].imshow(A[2,:].reshape((31, 23)), cmap='gray')
axes[0,3].imshow(A[3,:].reshape((31, 23)), cmap='gray')
axes[0,4].imshow(A[4,:].reshape((31, 23)), cmap='gray')
axes[1,0].imshow(B[0,:].reshape((31, 23)), cmap='gray')
axes[1,1].imshow(B[1,:].reshape((31, 23)), cmap='gray')
axes[1,2].imshow(B[2,:].reshape((31, 23)), cmap='gray')
axes[1,3].imshow(B[3,:].reshape((31, 23)), cmap='gray')
axes[1,4].imshow(B[4,:].reshape((31, 23)), cmap='gray')
plt.show()

U, s, Vt = np.linalg.svd(B)
fig, axes = plt.subplots(1, 5)
axes[0].imshow(Vt.T[0, :].reshape((31, 23)), cmap='gray')
axes[1].imshow(Vt.T[1, :].reshape((31, 23)), cmap='gray')
axes[2].imshow(Vt.T[2,:].reshape((31, 23)), cmap='gray')
axes[3].imshow(Vt.T[3, :].reshape((31, 23)), cmap='gray')
axes[4].imshow(Vt.T[4,:].reshape((31, 23)), cmap='gray')
plt.show()