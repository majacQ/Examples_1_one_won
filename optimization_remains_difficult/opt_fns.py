
import numpy as np
from scipy.optimize import minimize

b = 1e-6
c = 1e-3
E = 1e-2
G = 1e+3


def dsq(x: np.ndarray, *, i: int, j: int):
    """charge to charge distance squared as a function of x"""
    return (x[i] - x[j])**2 + (b * i - b * j)**2


def f(x) -> float:
    """objective to minimize"""
    x = np.array(x)
    k = x.shape[0]
    fx = 0
    for i in range(k):
        for j in range(k):
            if i != j:
                fx += E / ( 2 * np.sqrt(dsq(x, i=i, j=j)) )
        zi = c * np.abs(x[i] - i)
        fx += G * zi
    return fx


def g(x) -> np.ndarray:
    """gradient of objective to minimize"""
    x = np.array(x)
    k = x.shape[0]
    gx = np.zeros((k,), dtype=float)
    for i in range(k):
        for j in range(k):
            if i != j:
                gx[i] += -E * (x[i] - x[j]) / dsq(x, i=i, j=j)**(3/2)
        dzi = c * np.sign(x[i] - i)
        gx[i] += G * dzi
    return gx


def est_grad(x, *, epsilon: float = 1e-5):
    """numerical approximation of the gradient"""
    x = np.array(x, dtype=float)
    k = x.shape[0]
    fx = f(x)
    g_est = np.zeros((k,), dtype=float)
    for i in range(k):
        xi = x[i]
        x[i] = xi + epsilon
        fplus = f(x)
        x[i] = xi
        g_est[i] = (fplus - fx) / np.abs(epsilon)
    return g_est

