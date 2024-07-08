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
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

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

R0_vals = [1, 2, 3]
alpha_vals = [0.4]


N_max = 10  # Maximum value for N in the distribution

roots= np.zeros((len(alpha_vals), len(R0_vals)))
for i in range(len(alpha_vals)):
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    
    ax.set(xlim=(0, 1), ylim=(-0.2, 1))
    #plt.title(r"$u$ versus $G_{1}(u)$")
    ax.set_xlabel(r"Polynomial input, $u$")
    ax.set_ylabel(r"$G_{1}(u) - u$")
    axins = fig.add_axes([.55, .5, .3, .3])
    axins.tick_params(labelleft=True, labelbottom=True)
    axins.set(xlim=(0.6, 1), ylim=(-0.1, 0.05))

    for j in range(len(R0_vals)):
        alpha = alpha_vals[i]
        n = alpha
        R0 = R0_vals[j] # d is the dispersion parameter
        p = alpha / (R0 + alpha) #Changed this parameter
        dist = nbinom(n=n, p=p)  # Scipy parameterizes differently, check this
        my_pdf = dist.pmf(range(N_max + 1))
        my_pdf = my_pdf/np.sum(my_pdf)
        my_pgf_coef = make_G_u_minus_u(my_pdf)
        ax.plot(np.linspace(0, 1, 100), np.polyval(my_pgf_coef[::-1], np.linspace(0, 1, 100)), label = r"$R_{0}$ = %0.1f, $\alpha$ = %.1f"%(R0, alpha))
        axins.plot(np.linspace(0, 1, 100), np.polyval(my_pgf_coef[::-1], np.linspace(0, 1, 100)))
        roots[i,j] = _solve_self_consistent_equation(my_pdf)

ax.axhline(y = 0, color = 'grey', linestyle = '-')
axins.axhline(y = 0, color = 'grey', linestyle = '-')
ax.legend(loc = 'upper left')
plt.savefig("/Users/mcboudre/Documents/LSD_Lab/Thesi/solving_g1_of_u_r0_alpha_"+date+".pdf", format = "pdf")
plt.close()
# plt.show()

