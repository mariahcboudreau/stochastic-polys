import numpy as np
from numpy.linalg import norm, eig
from scipy.linalg import schur
from scipy.special import factorial2
import matplotlib.pyplot as plt
import random

# class PGF:
#     def __init__(self, coef):
#         self.coef = np.array(coef, dtype=float)

#     def __call__(self, x):
#         return sum(self.coef[i] * x**i for i in range(len(self.coef)))

#     def derivative(self):
#         deriv_coefs = self.coef[1:] * np.arange(1, len(self.coef))
#         return PGF(deriv_coefs)

#     def normalize(self):
#         """
#         Normalizes coefficients
#         """
#         self.coef/np.sum(self.coef)

def derivative(p_kCoeff):
    p_kCoeff = np.array(p_kCoeff)
    
    if p_kCoeff.size == 0:
        return np.array([0])
    elif p_kCoeff.size == 1:
        return np.array([0])
    else:
        primeCoeff = np.empty((len(p_kCoeff)-1))
        for p in range(len(p_kCoeff)-1):
            primeCoeff[p] = p_kCoeff[p+1]*(p+1)
        
    return primeCoeff 


def double_factorial(n):
    return factorial2(n - 1)


def omega(n):
    a1 = np.sqrt(2 / (np.pi * (n - 0.5)))
    a2 = np.sqrt((184 * n**4) + 23 * n + 23 * n**2 + 184 * n**4)
    return a1 * a2


def generate_sphere_point(dim):
    vec = np.random.randn(
        dim
    )  # Normalized random values from standard normal (mean = 0, stdev = 1)
    return vec / norm(vec)


def companion_matrix(coefs):
    coefs = np.array(coefs) # needs the coefficients as if htey are the highest to lowest exponents
                            # Constant last
    n = len(coefs)
    A = np.zeros((n - 1, n - 1))
    A[1:, :-1] = np.eye(n - 2)
    if coefs[0] != 0:
        A[0, :] = -coefs[1:n] / coefs[0]
    else:
        A[0, :] = -coefs[1:n]
    return A


# Retest polynomial_roots with corrected companion matrix function
def polynomial_roots(poly_coef):
    C = companion_matrix(poly_coef)
    return np.linalg.eigvals(C)


def in_bounds(coefs):
    """
    Check if the coefficients of a polynomial are within a specified range.

    Args:
        coefs (numpy.ndarray): The coefficients of the polynomial.
    Returns:
        numpy.ndarray: A boolean array indicating whether each coefficient is within the specified range.
    """
    epsilon = np.finfo(float).eps
    epsilon = 1e-6
    lower_bound = -epsilon
    upper_bound = 1.0 - epsilon
    real_coefs = np.real(coefs)
    # Check if real_coefs are greater than or equal to lower_bound
    lower_check = np.logical_or(real_coefs >= lower_bound, np.isclose(real_coefs, lower_bound, atol=epsilon))
    # Check if real_coefs are less than upper_bound
    upper_check = np.logical_or(real_coefs <= upper_bound, np.isclose(real_coefs, upper_bound, atol=epsilon, rtol=0))
    
    return np.logical_and(lower_check, upper_check)


def is_real(coefs):
    """
    Check if the given coefficients are real numbers.

    Parameters:
    coefs (array-like): The coefficients to be checked.

    Returns:
    bool: True if all coefficients are real, False otherwise.
    """
    return np.isclose(np.imag(coefs), 0)


def kappa_SCE(my_poly_coef, K=10, conditions=None, delta=0.001):
    if conditions is None:
        conditions = []
    SCE_list = []
    N = len(my_poly_coef)
    vec_list = [generate_sphere_point(N) for _ in range(K)]
    Z = np.column_stack(vec_list)

    for i in range(K):
        og_roots = polynomial_roots(my_poly_coef)
        all_conditions = np.array([True] * len(og_roots))
        if conditions:
            all_conditions = np.logical_and.reduce(
                [cond(og_roots) for cond in conditions]
            )

        og_roots = og_roots[all_conditions]
        delta = np.sqrt(norm(og_roots) * np.finfo(float).eps)
        perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])
        perturbed_roots = polynomial_roots(perturbed_coefs)[all_conditions]

        SCE_list.append(np.abs(og_roots - perturbed_roots) / delta * np.abs(og_roots))

    normed_sce = np.linalg.norm(SCE_list, axis=0)
    return np.mean(normed_sce)  # Simplified to return the mean as a scalar


# def make_G_u_minus_u(coefs):
#     # G_prime = G.derivative()
#     G = PGF(coefs)
#     G.normalize()
#     G_minus_u_coef = np.copy(G.coef)
#     G_minus_u_coef[1] -= 1
#     return G_minus_u_coef


# Test kappa_SCE with the polynomial coefficients for 2x^2 - 3x + 1
# test_kappa_sce = kappa_SCE([2, -3, 1], K=5)  # Reduced K for simplicity
# print(test_kappa_sce)
