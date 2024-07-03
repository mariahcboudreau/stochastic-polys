import numpy as np
from numpy.linalg import norm, eig
from scipy.linalg import schur
from scipy.special import factorial2
import matplotlib.pyplot as plt
import random

from stochastic_pgfs.pgfs import *


def double_factorial(n):
    return factorial2(n - 1)


# Approximation for the Wallis factor
def omega(n):
    a1 = np.sqrt(2 / (np.pi * (n - 0.5)))
    a2 = np.sqrt((184 * n**4) + 23 * n + 23 * n**2 + 184 * n**4)
    return a1 * a2


def generate_sphere_point(dim):
    vec = np.random.randn(dim)
    return vec / norm(vec)


def companion_matrix(coefs):
    coefs = np.array(coefs)
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
    return np.logical_and(0 <= np.real(coefs), np.real(coefs) < 0.999999999999)


def is_real(coefs):
    return np.isclose(np.imag(coefs), 0)


def l_x_metric(og_roots, perturbed_roots, delta, K, N):
    SCE_list = []
    SCE_list.append(np.abs(og_roots - perturbed_roots) / delta * np.abs(og_roots))
    normed_sce = np.linalg.norm(SCE_list, axis=0)

    return omega(K) / omega(N) * np.mean(normed_sce)


def l_x_algo(
    my_poly_coef,
    is_pgf=True,
    K=10,
    conditions=None,
    delta=0.001,
    perturbation_type="additive",
):
    if conditions is None:
        conditions = []
    all_og_roots_conditions = np.empty(K)
    all_perturbed_roots_conditions = np.empty(K)
    N = len(my_poly_coef)
    vec_list = [generate_sphere_point(N) for _ in range(K)]  # Random error
    Z = np.column_stack(vec_list)

    SCE_list = []

    # Root solving and error
    for i in range(K):
        if is_pgf:
            my_pgf_coef = make_G_u_minus_u(my_poly_coef)
            all_og_roots = polynomial_roots(my_pgf_coef)

            all_conditions = np.array([True] * len(all_og_roots))
            if conditions:  # Double checks they are roots
                all_conditions = np.logical_and.reduce(
                    [cond(all_og_roots) for cond in conditions]
                )
            og_roots = all_og_roots[all_conditions]
            delta = np.sqrt(norm(all_og_roots) * np.finfo(float).eps)
            # delta = np.sqrt(norm(all_og_roots) * 2**(-16))
            # delta = 10**(-8)
            #delta = 1
            if perturbation_type == "additive":
                perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])
            elif perturbation_type == "multiplicative":
                perturbed_coefs = my_poly_coef * (1 * delta * Z[:, i])
            else:
                assert "The perturbation type is not valid. Please choose 'additive' or 'multiplicative'"
            perturbed_coefs = make_G_u_minus_u(perturbed_coefs)

        else:
            all_og_roots = polynomial_roots(my_poly_coef)
            all_conditions = np.array([True] * len(og_roots))
            if conditions:  # Double checks they are roots
                all_conditions = np.logical_and.reduce(
                    [cond(og_roots) for cond in conditions]
                )
            og_roots = all_og_roots[all_conditions]

            if len(og_roots) < 1:
                assert("No roots found")
                return(np.nan)

            # delta = np.sqrt(norm(all_og_roots) * np.finfo(float).eps)
            # delta = 1
            if perturbation_type == "additive":
                perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])
            elif perturbation_type == "multiplicative":
                perturbed_coefs = my_poly_coef * (1 * delta * Z[:, i])
            else:
                assert "The perturbation type is not valid. Please choose 'additive' or 'multiplicative'"
        breakpoint()

        all_perturbed_roots = polynomial_roots(perturbed_coefs)
        all_conditions = np.array([True] * len(all_perturbed_roots))
        if conditions:  # Double checks they are roots
            all_conditions = np.logical_and.reduce(
                    [cond(all_perturbed_roots) for cond in conditions]
            )
        
        if len(all_perturbed_roots[all_conditions]) > 1:#if there are multiple roots meeting conditions, take the smallest one
            perturbed_roots = np.min(all_perturbed_roots[all_conditions])
        elif len(all_perturbed_roots[all_conditions]) == 0:#if no roots meet the conditions
            real_perturbed = all_perturbed_roots[np.isreal(all_perturbed_roots)]
            if real_perturbed[1] < 0:
                perturbed_roots = real_perturbed[1]
        else:
            perturbed_roots = all_perturbed_roots[all_conditions][0]
        #breakpoint() 
        SCE_list.append(np.abs(perturbed_roots - og_roots) / delta * np.abs(og_roots)) 
        ## Take the differences then divide by the original root value, percentage of original, and 

        # Both conditions perform this step after preprocessing
        # ONLY LOOK AT THIS FOR DEBUGGING
        all_perturbed_roots_conditions[i] = perturbed_roots

        all_og_roots_conditions[i] = og_roots[0]

    normed_sce = np.linalg.norm(SCE_list, axis=0) #provides the total displacement of all differences. 
    return normed_sce
    #return omega(K) / omega(N) * normed_sce
