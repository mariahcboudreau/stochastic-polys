from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from numba import jit, config
import logging
from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds, _solve_self_consistent_equation,G1_prime,get_outbreak_size
from stochastic_pgfs.random_graphs import poisson_degree_sequence, powerlaw_degree_sequence

def calculate_critical_transition(my_degree_sequence):
    pk = np.vstack((np.arange(0, my_degree_sequence.shape[0], 1), my_degree_sequence)).T
    return 1 / G1_prime(1, pk, 1)

N_max = 500
poisson_degree_sequence_partial = partial(poisson_degree_sequence, N_max=N_max)
powerlaw_degree_sequence_partial = partial(powerlaw_degree_sequence, N_max=N_max)


#my_degree_sequence = poisson_degree_sequence_partial(5)
T = np.linspace(0.001, 1, 200)
alpha_vals = np.linspace(1.8, 3.4, 5)
fig, ax = plt.subplots()
colors = plt.cm.viridis(np.linspace(0, 1,alpha_vals.shape[0]))
for idx, alpha in enumerate([1.8, 2.1, 3.0, 3.2]):
    my_degree_sequence = powerlaw_degree_sequence_partial(alpha)
    outbreak_size = [get_outbreak_size(my_degree_sequence, T_val,) for T_val in T]
    crit_trans = calculate_critical_transition(my_degree_sequence)
    ax.plot(T, outbreak_size, label=f'alpha = {alpha}', color=colors[idx])
    ax.axvline(crit_trans, color=colors[idx])
plt.legend()