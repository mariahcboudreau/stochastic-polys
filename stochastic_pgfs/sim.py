import numpy as np
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

from stochastic_pgfs.pgfs import *
from stochastic_pgfs.laub_xia_algo import *
from stochastic_pgfs.stochastic_poly import *

####
#
#   Class defining PGFs, sets up coefficients and determines derivatives
#
####

# class PGF:
#     def __init__(self, coef):
#         self.coef = np.array(coef, dtype=float)

#     def __call__(self, x):
#         return sum(self.coef[i] * x**i for i in range(len(self.coef)))

#     def derivative(self):
#         deriv_coefs = self.coef[1:] * np.arange(1, len(self.coef))
#         return PGF(deriv_coefs)


# def variance_polys(my_poly_coef, num_sims, K=10, conditions=None, delta=0.001):
#     if conditions is None:
#         conditions = []
#     SCE_list = []
#     N = len(my_poly_coef)
#     vec_list = [generate_sphere_point(N) for _ in range(K)]
#     Z = np.column_stack(vec_list)
#     all_perturbed_roots = np.zeros((num_sims, K))
#     for s in range(num_sims):
#         for i in range(K):
#             og_roots = polynomial_roots(my_poly_coef)
#             all_conditions = np.array([True] * len(og_roots))
#             if conditions:
#                 all_conditions = np.logical_and.reduce([cond(og_roots) for cond in conditions])
#             og_roots = og_roots[all_conditions]
#             delta = np.sqrt(norm(og_roots) * np.finfo(float).eps)
#             perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])
#             try:
#                 all_perturbed_roots[s,i] = polynomial_roots(perturbed_coefs)[all_conditions]
#             except:
#                 all_perturbed_roots[s,i] = 1

#     root_vars = all_perturbed_roots.var(axis = 1)
#     return root_vars.mean()  # Simplified to return the mean as a scalar

# Checked on 1/20/22 -
# Used in both winkler and winklerTrunc to as the conversion factor
######

####
# Checked on 1/20/22 - works
# Plot is not used as of now 1/20
#####


def violins(roots):

    df_roots = pd.DataFrame(roots)

    plt.figure(figsize=(7, 7))
    plt.title("PGF Sensitivity")
    plt.xlabel("Standard Deviation (times 0.01)")
    plt.ylabel("Root U")
    ax = sns.boxplot(data=df_roots, color="seagreen")
    plt.show(ax)


def neg_binom(r0, k, maxk):
    g1True = np.zeros(maxk)
    a = 1 / k
    for i in range(maxk):
        g1True[i] = (
            (math.gamma(i + k) / (math.factorial(i) * math.gamma(k)))
            * ((a * r0) / (1 + a * r0)) ** (i)
            * (1 / (1 + a * r0)) ** (k)
        )

    return g1True


def variance_sim(my_poly_coef, trials, conditions):

    sim_og_roots = np.zeros(trials)
    sim_pert_roots = np.zeros(trials)
    N = len(my_poly_coef)
    vec_list = [generate_sphere_point(N) for _ in range(trials)]  # Random error
    Z = np.column_stack(vec_list)

    for i in range(trials):

        my_pgf_coef = make_G_u_minus_u(my_poly_coef)
        og_roots = polynomial_roots(my_pgf_coef)

        all_conditions = np.array([True] * len(og_roots))
        if conditions:  # Double checks they are roots
            all_conditions = np.logical_and.reduce(
                [cond(og_roots) for cond in conditions]
            )
        og_roots = og_roots[all_conditions]
        delta = np.sqrt(norm(og_roots) * np.finfo(float).eps)
        # delta = 1

        ### Change to multiplicative noise
        perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])
        perturbed_coefs = make_G_u_minus_u(perturbed_coefs)
        perturbed_roots = polynomial_roots(perturbed_coefs)[all_conditions]

        sim_og_roots[i] = np.real(og_roots[0])
        sim_pert_roots[i] = np.real(perturbed_roots)[0]

    var_of_all = np.var(sim_pert_roots, axis=0)
    # violins(perturbed_coefs)
    # ridgePlots(U)
    return var_of_all


def variance_sim_multiplicative(my_poly_coef, trials, conditions):

    sim_og_roots = np.zeros(trials)
    sim_pert_roots = np.zeros(trials)
    N = len(my_poly_coef)
    vec_list = [generate_sphere_point(N) for _ in range(trials)]  # Random error
    Z = np.column_stack(vec_list)

    for i in range(trials):

        my_pgf_coef = make_G_u_minus_u(my_poly_coef)
        og_roots = polynomial_roots(my_pgf_coef)

        all_conditions = np.array([True] * len(og_roots))
        if conditions:  # Double checks they are roots
            all_conditions = np.logical_and.reduce(
                [cond(og_roots) for cond in conditions]
            )
        og_roots = og_roots[all_conditions]
        delta = np.sqrt(norm(og_roots) * np.finfo(float).eps)
        # delta = 1

        ### Change to multiplicative noise
        perturbed_coefs = my_poly_coef * (delta * Z[:, i])
        perturbed_coefs = make_G_u_minus_u(perturbed_coefs)
        perturbed_roots = polynomial_roots(perturbed_coefs)[all_conditions]

        sim_og_roots[i] = np.real(og_roots[0])
        sim_pert_roots[i] = np.real(perturbed_roots)[0]

    var_of_all = np.var(sim_pert_roots, axis=0)
    # violins(perturbed_coefs)
    # ridgePlots(U)
    return var_of_all
