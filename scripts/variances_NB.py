import os
import sys
sys.path.insert(1,os.getcwd())

from stochastic_pgfs.laub_xia_algo import l_x_algo, iterate_with_acceleration, G1, _perturb_polynomial
import numpy as np 
from stochastic_pgfs.sim import *
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
date = datetime.today().strftime('%m-%d-%Y')



# Log space values 
# k_vals = [8, 0.3, 0.3, 0.3, 3, 0.003]

# # R0_vals = np.linspace(0.5,4,10)
# R0_vals = [1.05, 1.2, 2, 4, 2, 2]


k_vals = np.linspace(0.01, 10, 175)

tracked_k = np.array([k_vals[2], k_vals[140]]) # 1 - 0.06 2 - 0.1248 4 - 0.24 53 - 3.05, 140 - around 8

# R0_vals = np.linspace(0.5,4,10)
R0_vals = np.arange(0.8, 4.2, 0.2)

tracked_R0 = np.array([R0_vals[2], R0_vals[3], R0_vals[15]]) # 2 - 1.2 3 - 1.4, 14 - 3.8

N_max = 1000 # Maximum value for N in the distribution


variances = np.zeros((len(k_vals), len(R0_vals)))
perturbed = np.zeros((len(tracked_k), len(tracked_R0), int(1e4)))
og_roots = np.zeros((len(tracked_R0), len(tracked_R0)))

delta = np.float64(2**(-2))



def variance_calc(my_poly_coef, delta, K = int(1e4), save_dist = False):
    
    T = 1
    tol=1e-10
    acceleration_method='steffensen'
    perturbation_type="additive"


    my_poly_coef= my_poly_coef/np.arange(1,len(my_poly_coef)+1)
    my_poly_coef = np.insert(my_poly_coef, 0, 0)

    my_poly_coef = np.ascontiguousarray(my_poly_coef)
    my_poly_coef = np.vstack((np.arange(0, my_poly_coef.shape[0], 1), my_poly_coef)).T.copy()

    N = my_poly_coef.shape[0]

    Z = np.empty((N, K), dtype=np.float64)
    for i in range(K):
        vec = np.random.randn(N)
        Z[:, i] = vec / np.linalg.norm(vec)

    og_roots, _ = iterate_with_acceleration(
        my_poly_coef, 
        T=T, 
        method=acceleration_method,
        tol=tol
    )

    perturbed_roots_list = np.zeros((K), dtype=np.float64)    

    for i in range(K):
        if i % 1000 == 0:
            logging.info(f"Processing perturbation {i}/{K}")
            
        alpha_i = Z[:, i]
        my_perturbed_poly_coefs = _perturb_polynomial(
            my_poly_coef.copy(), 
            delta, 
            alpha_i, 
            perturbation_type
        )
        
        perturbed_roots, _ = iterate_with_acceleration(
            my_perturbed_poly_coefs,
            T=T,
            method=acceleration_method,
            tol=tol,
            usol=og_roots
        )
       
        perturbed_roots_list[i] = perturbed_roots

    if save_dist:
        return np.var(perturbed_roots_list), perturbed_roots_list, og_roots
    else:
        return np.var(perturbed_roots_list)





for i in range(len(k_vals)):
        for j in range(len(R0_vals)):
            save_dist = False
            k = k_vals[i]
            n = k
            R0 = R0_vals[j] 
            p = k / (R0 + k) 
            dist = nbinom(n=n, p=p)  # Scipy parameterizes with alpha / R0 + alpha
            my_pdf = dist.pmf(range(N_max + 1))
            my_pdf = my_pdf/np.sum(my_pdf)
            if (k in tracked_k) and (R0 in tracked_R0):
                variances[i,j], perturbed[np.where(tracked_k== k)[0], np.where(tracked_R0== R0)[0], :], og_roots[np.where(tracked_k== k)[0], np.where(tracked_R0== R0)[0]] = variance_calc(my_pdf, delta, K = int(1e4), save_dist = True)
                
            else:
                variances[i,j] = variance_calc(my_pdf, delta, K = int(1e4), save_dist = False)





# X, Y = np.meshgrid(R0_vals, k_vals)
# # fig, ax = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)

# plt.imshow(variances,
#             extent = (min(R0_vals),max(R0_vals),min(k_vals),max(k_vals)),
#              origin = 'lower', aspect = 'auto', cmap = 'pink_r'
#             )
# plt.xlabel( r'Average secondary cases, $R_0$')
# plt.ylabel(ylabel=r'Dispersion parameter, $k$')
# plt.yscale('log')
# plt.title('Variances')
# # plt.show()
# plt.savefig("/Users/mar.boudreau/Documents/UVM/stochastic-polys/figures/variance_heatmaps.pdf", format = "pdf")
# plt.close()

perturbed_df = pd.DataFrame()
for i in range(len(tracked_k)):
        for j in range(len(tracked_R0)):
            perturbed_df['[%.2f, %.2f]'%(tracked_R0[j], tracked_k[i])] = perturbed[i,j,:]

data_centered = perturbed_df - perturbed_df.mean()

fig, axes = plt.subplots()

axes.violinplot(dataset = [data_centered["[1.20, 8.05]"].values,
                           data_centered["[1.40, 0.12]"].values,
                           data_centered["[3.80, 0.12]"].values] )

# data_centered.boxplot(grid = False, column = ["[1.20, 8.05]", "[1.40, 0.12]", "[3.80, 0.12]" ])
plt.xlabel(r'$R_{0}$ and $k$ pairs')
plt.ylabel('Relative variance')
plt.tight_layout()
plt.savefig('/Users/mar.boudreau/Documents/UVM/stochastic-polys/figures/variance_boxplot_02-11-2025.pdf', format = 'pdf')
# plt.show()

print(perturbed_df)
print(data_centered)