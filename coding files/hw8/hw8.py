import matplotlib.pyplot as plt
import numpy as np
import q5

with np.load(r'faces.npz') as data:
    A = data['A']

plt.imshow(A[0,:].reshape((31, 23)), cmap='gray')
plt.show()

### Question 1
# find the mean of all 766 pictures
mean = np.mean(A, 0)
plt.imshow(np.reshape(mean, (31, 23)), cmap='gray')
plt.show()

### Question 2
B = np.subtract(A, mean)

### Question 3
U, s, Vt = np.linalg.svd(B)
log_s = np.log(s)

plt.plot(log_s)
plt.xlabel('j')
plt.ylabel('log(s_j)')
plt.show()

# Question 4
fig, axes = plt.subplots(1, 5)

for i in range(5):
    row = Vt[i, :]
    axes[i].imshow(np.reshape(row, (31, 23)), cmap='gray')
    axes[i].set_title("right singular vector {} of B".format(i+1))

fig.tight_layout()
plt.show()

# Question 5
r_30, B_30 = q5.compute_error(B, U, s, Vt, 0.3)
r_20, B_20 = q5.compute_error(B, U, s, Vt, 0.2)
r_10, B_10 = q5.compute_error(B, U, s, Vt, 0.1)
r_1, B_1 = q5.compute_error(B, U, s, Vt, 0.01)

print(r_30, r_20, r_10, r_1)

# Question 6
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)

ax1.imshow(np.reshape(B[0,:], (31,23)), cmap='gray')
ax1.set_title("B original")
ax2.imshow(np.reshape(B_1[0,:], (31,23)), cmap='gray')
ax2.set_title("B_r, r=546, error=1%")
ax3.imshow(np.reshape(B_10[0,:], (31,23)), cmap='gray')
ax3.set_title("B_r, r=207, error=10%")
ax4.imshow(np.reshape(B_20[0,:], (31,23)), cmap='gray')
ax4.set_title("B_r, r=101, error=20%")
ax5.imshow(np.reshape(B_30[0,:], (31,23)), cmap='gray')
ax5.set_title("B_r, r=53, error=30%")

fig.tight_layout()
plt.show()