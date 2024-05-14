import numpy as np
from numpy.linalg import norm, eig
from scipy.linalg import schur
from scipy.special import factorial2
import scipy
import matplotlib.pyplot as plt
import random

from stochastic_pgfs.pgfs import *
from stochastic_pgfs.laub_xia_algo import kappa_SCE, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.stochastic_poly import *

####
#
#   Class defining PGFs, sets up coefficients and determines derivatives
#
####

class PGF:
    def __init__(self, coef):
        self.coef = np.array(coef, dtype=float)
    
    def __call__(self, x):
        return sum(self.coef[i] * x**i for i in range(len(self.coef)))
    
    def derivative(self):
        deriv_coefs = self.coef[1:] * np.arange(1, len(self.coef))
        return PGF(deriv_coefs)
    

def variance_polys(my_poly_coef, num_sims, K=10, conditions=None, delta=0.001):
    if conditions is None:
        conditions = []
    SCE_list = []
    N = len(my_poly_coef)
    vec_list = [generate_sphere_point(N) for _ in range(K)]
    Z = np.column_stack(vec_list)
    all_perturbed_roots = np.zeros((num_sims, K))
    for s in range(num_sims):
        for i in range(K):
            og_roots = polynomial_roots(my_poly_coef)
            all_conditions = np.array([True] * len(og_roots))
            if conditions:
                all_conditions = np.logical_and.reduce([cond(og_roots) for cond in conditions])
            og_roots = og_roots[all_conditions]
            delta = np.sqrt(norm(og_roots) * np.finfo(float).eps)
            perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])
            all_perturbed_roots[s,i] = polynomial_roots(perturbed_coefs)[all_conditions][0]
    
    root_vars = all_perturbed_roots.var(axis = 1)
    return root_vars.mean()  # Simplified to return the mean as a scalar
