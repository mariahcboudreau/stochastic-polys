import numpy as np
from numpy.linalg import norm, eig
from scipy.linalg import schur
from scipy.special import factorial2
import matplotlib.pyplot as plt
import random

from stochastic_pgfs.pgfs import *

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



def double_factorial(n):
    return factorial2(n-1)

# Approximation for the Wallis factor
def omega(n):
    a1 = np.sqrt(2 / (np.pi * (n - 0.5)))
    a2 = np.sqrt((184*n**4) + 23*n + 23*n**2 + 184*n**4)
    return a1 * a2

def generate_sphere_point(dim):
    vec = np.random.randn(dim)
    return vec / norm(vec)

def companion_matrix(coefs):
    coefs = np.array(coefs)
    n = len(coefs)
    A = np.zeros((n-1, n-1))
    A[1:, :-1] = np.eye(n-2)
    A[0, :] = -coefs[1:n] / coefs[0]
    return A

# Retest polynomial_roots with corrected companion matrix function
def polynomial_roots(poly_coef):
    C = companion_matrix(poly_coef)
    return np.linalg.eigvals(C)

def in_bounds(coefs):
    return np.logical_and(0 < np.real(coefs), np.real(coefs) <= 1)

def is_real(coefs):
    return np.isclose(np.imag(coefs), 0)


def l_x_algo(my_poly_coef, is_pgf, K = 10, conditions = None, delta = 0.001):
    if conditions is None:
        conditions = []
    all_og_roots = np.empty(K)
    all_perturbed_roots = np.empty(K)
    N = len(my_poly_coef)
    vec_list = [generate_sphere_point(N) for _ in range(K)] # Random error
    Z = np.column_stack(vec_list)

    SCE_list = []
    
    
    # Root solving and error
    for i in range(K):
        if is_pgf:
            my_pgf_coef = make_G_u_minus_u(my_poly_coef)
            og_roots = polynomial_roots(my_pgf_coef)

            all_conditions = np.array([True] * len(og_roots))
            if conditions: #Double checks they are roots
                all_conditions = np.logical_and.reduce([cond(og_roots) for cond in conditions])
            og_roots = og_roots[all_conditions]
            delta = np.sqrt(norm(og_roots) * np.finfo(float).eps)

            perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])
            perturbed_coefs = make_G_u_minus_u(perturbed_coefs)
            

        else:
            og_roots = polynomial_roots(my_poly_coef)
            all_conditions = np.array([True] * len(og_roots))
            if conditions: #Double checks they are roots
                all_conditions = np.logical_and.reduce([cond(og_roots) for cond in conditions])
            og_roots = og_roots[all_conditions]
            delta = np.sqrt(norm(og_roots) * np.finfo(float).eps)
            perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])

        perturbed_roots = polynomial_roots(perturbed_coefs)[all_conditions][0]
        SCE_list.append(np.abs(og_roots - perturbed_roots) / delta * np.abs(og_roots))
    
    

        # Both conditions perform this step after preprocessing
        all_perturbed_roots[i] = np.min(np.real(perturbed_roots))

        all_og_roots[i] = np.min(np.real(og_roots))

    normed_sce = np.linalg.norm(SCE_list, axis=0)
    return np.mean(normed_sce)

# def kappa_SCE(my_poly_coef, is_pgf, K=10, conditions=None, delta=0.001 ):
    # if conditions is None:
    #     conditions = []
    
    # N = len(my_poly_coef)
    # vec_list = [generate_sphere_point(N) for _ in range(K)] # Random error
    # Z = np.column_stack(vec_list)
    
    # for i in range(K):
    #     og_roots = polynomial_roots(my_poly_coef)
    #     all_conditions = np.array([True] * len(og_roots))
    #     if conditions: #Double checks they are roots
    #         all_conditions = np.logical_and.reduce([cond(og_roots) for cond in conditions])
        
    #     og_roots = og_roots[all_conditions]
    #     delta = np.sqrt(norm(og_roots) * np.finfo(float).eps)
    #     perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])
    #     perturbed_roots = polynomial_roots(perturbed_coefs)[all_conditions]
    # og_roots, perturbed_roots = l_x_algo(my_poly_coef, is_pgf, K)
    
    # return np.mean(normed_sce)  # Simplified to return the mean as a scalar




# # Test kappa_SCE with the polynomial coefficients for 2x^2 - 3x + 1
test_kappa_sce = l_x_algo([2, -3, 1], is_pgf = False, K = 10, conditions=[is_real, in_bounds])  # Reduced K for simplicity
print(test_kappa_sce) ### Check this out. 
