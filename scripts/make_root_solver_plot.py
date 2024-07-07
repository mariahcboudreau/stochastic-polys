from stochastic_pgfs.laub_xia_algo import polynomial_roots, is_real, in_bounds, _solve_self_consistent_equation
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.viz_utils import outbreak_contours
import numpy as np
from stochastic_pgfs.sim import *
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
date = datetime.today().strftime('%m-%d-%Y')

def pgf_solver(my_poly_coef, conditions, solve_root = True):

    if np.sum(derivative(my_poly_coef)) > 1:
        my_pgf_coef = make_G_u_minus_u(my_poly_coef)
        all_og_roots = polynomial_roots(np.flip(my_pgf_coef))
        all_conditions = np.array([True] * len(all_og_roots))
        if conditions:  # Double checks they are roots
            all_conditions = np.logical_and.reduce(
                [cond(all_og_roots) for cond in conditions]
            )
        if len(all_og_roots[all_conditions]) > 1:
            og_roots = np.max(all_og_roots[all_conditions])
        elif len(all_og_roots[all_conditions]) == 0:
            og_roots = 1
        else:
            og_roots = all_og_roots[all_conditions][0]
    else:
        og_roots = 1
        my_pgf_coef = make_G_u_minus_u(my_poly_coef)
        all_og_roots = polynomial_roots(np.flip(my_pgf_coef))
        all_conditions = np.array([True] * len(all_og_roots))
        if conditions:  # Double checks they are roots
            all_conditions = np.logical_and.reduce(
                [cond(all_og_roots) for cond in conditions]
            )

    if solve_root:
        return og_roots
    else:
        return 1 - og_roots

alpha_vals = [1.5, 2, 2.5, 3]
R0_vals = [0.25, 0.75]


N_max = 10  # Maximum value for N in the distribution


def root_plotter(solver, alpha_vals, R0_vals, N_max):
    roots= np.zeros((len(alpha_vals), len(R0_vals)))
    for i in range(len(alpha_vals)):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.axis([0,3,0,1])
        plt.title(r"$G_{1}(u) versus u")
        plt.xlabel(r"Polynomial root, $u$")
        plt.ylabel(r"$G_{1}(u)$")
        for j in range(len(R0_vals)):
            alpha = alpha_vals[i]
            n = alpha
            R0 = R0_vals[j] # d is the dispersion parameter
            p = alpha / (R0 + alpha) #Changed this parameter
            dist = nbinom(n=n, p=p)  # Scipy parameterizes differently, check this
            my_pdf = dist.pmf(range(N_max + 1))
            my_pdf = my_pdf/np.sum(my_pdf)
            plt.plot(my_pdf)
            roots[i,j] = solver(my_pdf)
    return roots

