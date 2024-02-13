"""
Minh Phan
Homwork 3 Coding Part
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import compute_angle, angle_data, build_matrix
from gram_schmidt import gs


### Part I

# # set values for d and N
d = 2**4

N = 2**7
V1 = np.zeros([N, d])

np.random.seed(349573)
for i in range(V1.shape[0]):
    V1[i, :] = np.random.randn(d)

data1 = angle_data(V1)

N = 2**8
V2 = np.zeros([N, d])

np.random.seed(354981)
for i in range(V2.shape[0]):
    V2[i, :] = np.random.randn(d)

data2 = angle_data(V2)

N = 2**9
V3 = np.zeros([N, d])

np.random.seed(748343)
for i in range(V3.shape[0]):
    V3[i, :] = np.random.randn(d)

data3 = angle_data(V3)


# # plot the data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
sns.histplot(data1, ax=ax1, legend=False, bins=100).set(xlabel='angle (rad)')
sns.histplot(data2, ax=ax2, legend=False, bins=100).set(xlabel='angle (rad)')
sns.histplot(data3, ax=ax3, legend=False, bins=100).set(xlabel='angle (rad)')
ax1.set_title('N=2^7, d=2^4')
ax2.set_title('N=2^8, d=2^4')
ax3.set_title('N=2^9, d=2^4')
fig.tight_layout()
plt.show()


### Part 2

N = 2**7
V4 = np.zeros([N, N])
N = 2**8
V5 = np.zeros([N, N])
N = 2**9
V6 = np.zeros([N, N])

# build matrices of b_i's 
build_matrix(V4)
data4 = angle_data(V4)
build_matrix(V5)
data5 = angle_data(V5)
build_matrix(V6)
data6 = angle_data(V6)


# plot the data
fig, (ax4, ax5, ax6) = plt.subplots(1, 3)
sns.histplot(data4, ax=ax4, legend=False, bins=100).set(xlabel='angle (rad)')
sns.histplot(data5, ax=ax5, legend=False, bins=100).set(xlabel='angle (rad)')
sns.histplot(data6, ax=ax6, legend=False, bins=100).set(xlabel='angle (rad)')
ax3.set_title('N=2^7')
ax4.set_title('N=2^8')
ax5.set_title('N=2^9')
fig.tight_layout()
plt.show()


# collect pairwise angles between the qi's for different values of N
data7 = angle_data(gs(V4))
data8 = angle_data(gs(V5))
data9 = angle_data(gs(V6))

# build histogram
fig, (ax7, ax8, ax9) = plt.subplots(1, 3)
sns.histplot(data7, ax=ax7, legend=False, bins=100).set(xlabel='angle (rad)')
sns.histplot(data8, ax=ax8, legend=False, bins=100).set(xlabel='angle (rad)')
sns.histplot(data9, ax=ax9, legend=False, bins=100).set(xlabel='angle (rad)')
ax7.set_title('N=2^7')
ax8.set_title('N=2^8')
ax9.set_title('N=2^9')
fig.tight_layout()
plt.show()