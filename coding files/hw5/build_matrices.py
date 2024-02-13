import numpy as np
import math

def construct_equation(x):
    V = np.vander(x)

    f = lambda x: 1/(1+25*(x**2))
    b = [f(i) for i in x]

    return V, b


def construct_trig(array):
    N = len(array)

    def build_row(x):
        cos_func = lambda k, x: math.cos(k*x*math.pi)
        sin_func = lambda k, x: math.sin((k - N/2+1)*math.pi*x)
        
        result = [cos_func(k, x) for k in range(N//2)]
        result.extend([sin_func(k, x) for k in range(N//2, N)])
        return result
    
    V = []
    for x in array:
        V.append(build_row(x))
    f = lambda x: 1/(1+25*(x**2))
    b = [f(i) for i in array]

    return np.array(V), b


def interpolate_trig(grid, coeff):
    """
    returns the y coordinates of a trig function
    """
    N = len(coeff)
    y = []
    for x in grid:
        value = sum([coeff[k] * math.cos(k*math.pi*x) for k in range(N//2)])
        value += sum([coeff[k] * math.sin((k-N//2+1)*math.pi*x) for k in range(N//2, N)])
        y.append(value)
    
    return y

