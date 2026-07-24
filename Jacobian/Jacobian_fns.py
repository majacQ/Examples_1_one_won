

# https://terrytao.wordpress.com/2026/07/21/a-digestion-of-the-jacobian-conjecture-counterexample/

import itertools
import numpy as np
import sympy as sp


def map_f(fn, *, point, variables):
    """Evaluate polynomial map"""
    return tuple(fi.xreplace({zi: pi for zi, pi in zip(variables, point)}) for fi in fn)


def q_J_check(f, *, pts, variables) -> bool:
    """Check if determinant has the same value a few places"""
    J = sp.Matrix(f).jacobian(variables)
    d_set = set()
    saw_zero = False
    for pti in pts:
        di = J.xreplace({zi: pi for zi, pi in zip(variables, pti)}).det()
        if di==0:
            saw_zero = True
        d_set.add(di)
        if len(d_set) > 1:
            return False
    # len(d_set) >= 1
    if saw_zero:
        return False  # was only zero
    return True


def compute_detJ(F, *, variables):
    """Compute determinate of Jacobian"""
    # Vector function and its Jacobian (3x3)
    J = sp.Matrix(F).jacobian(variables)
    # Determinant of the Jacobian
    detJ = sp.simplify(J.det())
    return detJ


def solve_for_third_poly(*, mc, f1, f2, variables, determinant_target):
    ws = sp.symbols(",".join(["w_{" + str(v) + "}" for v in mc]))
    f_abstract = sum([wi * ti for wi, ti in zip(ws, mc)])
    J = sp.Matrix((f1, f2, f_abstract)).jacobian(variables)
    detJ = sp.expand(J.det())
    detJz = detJ.subs({v: 0 for v in variables})
    if detJz == 0:
        return None
    detJp = sp.Poly(detJ - determinant_target, variables)
    # monoms = detJp.monoms()        # tuples of exponents, e.g. (2,1,0)
    coeffs = detJp.coeffs()        # corresponding coefficients
    # extract linear relations
    eqns = [detJz - determinant_target]
    for coeff in coeffs:
        eqns.append(coeff)
    s3 = f_abstract.subs(sp.solve(eqns, ws)).subs({w: 0 for w in ws}).expand()
    if s3 == 0:
        return None 
    return s3


def find_repeated_roots(*, fn, num_bound: int, den_bound: int, variables):
    """Search for repeated roots"""
    root_candidates = tuple(sorted(set([
        sp.Rational(num, den) for num in range(-num_bound, num_bound + 1) for den in range(1, den_bound)])))
    mp = dict()
    for p in itertools.product(root_candidates, repeat=3):
        v = map_f(fn=fn, point=p, variables=variables)
        try:
            pre = mp[v]
        except KeyError:
            pre = []
        pre.append(p)
        mp[v] = pre
    max_arity = np.max([len(v) for v in mp.values()])
    mp = {k: v for k, v in mp.items() if (len(v) > 1) and (len(v) >= max_arity)}
    if len(mp) > 1:
        k = list(mp.keys())[0]
        mp = {k: mp[k]}
    return mp



def monomial_candidates(*, power_bound: int, deg_bound: int, variables):
    """Generate monomials"""
    candidates = set()
    for deg in range(1, deg_bound):
        for exps in itertools.product(range(power_bound), repeat=3):
            term = 1
            deg = 0
            for v, p in zip(variables, exps):
                if (p > 0) and (p < deg_bound):
                    deg += p
                    term *= v**p
            if (deg > 0) and (deg < deg_bound):
                candidates.add(term)
    return tuple(candidates)


def random_poly(*, rng, mc, term_bound: int, coef_bound: int):
    """Generate a random polynomial (not zero, and no constant term)"""
    res = 0
    while res == 0:
        n_terms = rng.integers(low=1, high=term_bound)
        for term in rng.choice(mc, n_terms, replace=False):
            coef = 0
            while coef == 0:
                coef = rng.integers(low=-(coef_bound - 1), high=coef_bound)
            res = res + coef * term
    return res



def check_candidate(Fcandidate, *, num_bound: int, den_bound: int, variables) -> bool:
    """Check if we have the right fn"""
    dJ = compute_detJ(Fcandidate, variables=variables)
    if (dJ != 0) and (len(dJ.free_symbols) == 0):
        rmap = find_repeated_roots(fn=Fcandidate, num_bound=num_bound, den_bound=den_bound, variables=variables)
        if len(rmap) > 0:
            return True
    return False

