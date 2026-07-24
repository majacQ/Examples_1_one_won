
# https://terrytao.wordpress.com/2026/07/21/a-digestion-of-the-jacobian-conjecture-counterexample/


import numpy as np
import sympy as sp

from multiprocessing import Pool
from Jacobian_fns import compute_detJ, find_repeated_roots, q_J_check, monomial_candidates, random_poly, solve_for_third_poly


z_1, z_2, z_3 = sp.symbols('z_1, z_2, z_3')
variables = (z_1, z_2, z_3)
num_bound = 8
den_bound = 4
power_bound = 6
deg_bound = 10
power_bound_f = 7
deg_bound_f = 11
coef_bound = 13
term_bound = 7
determinant_target = 1
mc = monomial_candidates(power_bound=power_bound_f, deg_bound=deg_bound_f, variables=variables)
rc = monomial_candidates(power_bound=power_bound, deg_bound=deg_bound, variables=variables)


def check_i(seed):
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(100):
        c1 = random_poly(term_bound=term_bound, coef_bound=coef_bound, mc=rc, rng=rng) + z_1
        c2 = random_poly(term_bound=term_bound, coef_bound=coef_bound, mc=rc, rng=rng) + z_2
        c3 = solve_for_third_poly(f1=c1, f2=c2, mc=mc, variables=variables, determinant_target=determinant_target)
        if (c3 is not None) and (c3 != 0):
            c = (c1, c2, c3)
            if q_J_check(c, pts=((0,0,0), (1,1,1)), variables=variables):
                dJ = compute_detJ(c, variables=variables)
                if (dJ is not None) and (dJ != 0) and (len(dJ.free_symbols) == 0):
                    rr = find_repeated_roots(fn=c, num_bound=num_bound, den_bound=den_bound, variables=variables)
                    print("found:", flush=True)
                    print(c, flush=True)
                    print("-", flush=True)
                    print(rr, flush=True)
                    print(".", flush=True)
                    found.append(c)
    if len(found) > 0:
        return found
    return None


if __name__ == "__main__":
    rng_a = np.random.default_rng(2026)
    candidates = [rng_a.integers(low=1, high=2**63) for _ in range(1000)]
    print("start", flush=True)
    with Pool(processes=6) as pool:
        results = pool.map(check_i, candidates)
    print("done", flush=True)
    res = [r for r in results if r is not None]
    print(res, flush=True)
