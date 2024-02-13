import numpy as np
import build_matrices
import matplotlib.pyplot as plt
import math

### Part 1
N= 2**3
x = np.linspace(-1, 1, N)
V1, b1 = build_matrices.construct_equation(x)
a1 = np.linalg.solve(V1, b1)

N= 2**4
x = np.linspace(-1, 1, N)
V2, b2 = build_matrices.construct_equation(x)
a2 = np.linalg.solve(V2, b2)

N= 2**5
x = np.linspace(-1, 1, N)
V3, b3 = build_matrices.construct_equation(x)
a3 = np.linalg.solve(V3, b3)

grid = np.linspace(-1, 1, 100)
f = lambda x: 1/(1+25*(x**2))
y = [f(x) for x in grid]

y1 = [np.polyval(a1, x) for x in grid]
y2 = [np.polyval(a2, x) for x in grid]
y3 = [np.polyval(a3, x) for x in grid]

plt.plot(grid, y, color='k', label='f')
plt.plot(grid, y1, color='c', label='N = 2^3')
plt.plot(grid, y2, color='m', label='N = 2^4')
plt.plot(grid, y3, color='y', label='N = 2^5')

plt.ylim([-0.5, 1.1])
plt.legend()
plt.show()

d1 = np.linalg.det(V1)
d2 = np.linalg.det(V2)
d3 = np.linalg.det(V3)

print(d1, d2, d3)

# Use Chebyshev nodes
cheb_func = lambda j, N: math.cos((2*(j+1)-1)/(2*N)*math.pi)

N = 2**3
x = [cheb_func(j, N) for j in range(N)]
V1, b1 = build_matrices.construct_equation(x)
a1 = np.linalg.solve(V1, b1)

N = 2**4
x = [cheb_func(j, N) for j in range(N)]
V2, b2 = build_matrices.construct_equation(x)
a2 = np.linalg.solve(V2, b2)

N = 2**5
x = [cheb_func(j, N) for j in range(N)]
V3, b3 = build_matrices.construct_equation(x)
a3 = np.linalg.solve(V3, b3)

y1 = [np.polyval(a1, x) for x in grid]
y2 = [np.polyval(a2, x) for x in grid]
y3 = [np.polyval(a3, x) for x in grid]


plt.plot(grid, y, color='k', label='f')
plt.plot(grid, y1, color='r', label='N=2^3')
plt.plot(grid, y2, color='g', label='N=2^4')
plt.plot(grid, y3, color='b', label='N=2^5')

plt.ylim([-0.5, 1.1])
plt.legend()
plt.show()

d1 = np.linalg.det(V1)
d2 = np.linalg.det(V2)
d3 = np.linalg.det(V3)

print(d1, d2, d3)

### Part 2

N= 2**3
x = np.linspace(-1, 1, N)
A1, b1 = build_matrices.construct_trig(x)
a1 = np.linalg.solve(A1, b1)

N= 2**4
x = np.linspace(-1, 1, N)
A2, b2 = build_matrices.construct_trig(x)
a2 = np.linalg.solve(A2, b2)

N= 2**5
x = np.linspace(-1, 1, N)
A3, b3 = build_matrices.construct_trig(x)
a3 = np.linalg.solve(A3, b3)

grid = np.linspace(-1, 1, 100)
f = lambda x: 1/(1+25*(x**2))
y = [f(x) for x in grid]

y1 = build_matrices.interpolate_trig(grid, a1)
y2 = build_matrices.interpolate_trig(grid, a2)
y3 = build_matrices.interpolate_trig(grid, a3)

plt.plot(grid, y, color='k', label='f')
plt.plot(grid, y1, color='c', label='N=2^3')
plt.plot(grid, y2, color='m', label='N=2^4')
plt.plot(grid, y3, color='y', label='N=2^5')

plt.ylim([-0.5, 1.1])
plt.legend()
plt.show()

d1 = np.linalg.det(A1)
d2 = np.linalg.det(A2)
d3 = np.linalg.det(A3)

print(d1, d2, d3)

# Use Chebyshev nodes
cheb_func = lambda j, N: math.cos((2*(j+1)-1)/(2*N)*math.pi)

N = 2**3
x = [cheb_func(j, N) for j in range(N)]
A1, b1 = build_matrices.construct_trig(x)
a1 = np.linalg.solve(A1, b1)

N = 2**4
x = [cheb_func(j, N) for j in range(N)]
A2, b2 = build_matrices.construct_trig(x)
a2 = np.linalg.solve(A2, b2)

N = 2**5
x = [cheb_func(j, N) for j in range(N)]
A3, b3 = build_matrices.construct_trig(x)
a3 = np.linalg.solve(A3, b3)

y1 = build_matrices.interpolate_trig(grid, a1)
y2 = build_matrices.interpolate_trig(grid, a2)
y3 = build_matrices.interpolate_trig(grid, a3)

plt.plot(grid, y, color='k', label='f')
plt.plot(grid, y1, color='r', label='N=2^3')
plt.plot(grid, y2, color='g', label='N=2^4')
plt.plot(grid, y3, color='b', label='N=2^5')

plt.ylim([-0.5, 1.1])
plt.legend()
plt.show()

d1 = np.linalg.det(A1)
d2 = np.linalg.det(A2)
d3 = np.linalg.det(A3)

print(d1, d2, d3)