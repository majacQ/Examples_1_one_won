
import numpy as np
from opt_fns import g, est_grad


def test_gradient():
    # We confirm we haven't made too many errors in the gradient formula and code.
    x = [0.95, 2.2, 0.3]
    grad = g(x)
    g_est = est_grad(x)
    assert np.max(np.abs(grad - g_est)) < 1e-6
