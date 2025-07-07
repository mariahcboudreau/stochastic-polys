import os
import sys
sys.path.insert(1,os.getcwd())

from stochastic_pgfs.laub_xia_algo import l_x_algo
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.viz_utils import outbreak_contours, condition_number_heatmap 
import numpy as np 
from stochastic_pgfs.sim import *
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import pyarrow
import jax
import jax.numpy as jnp
import stats
date = datetime.today().strftime('%m-%d-%Y')

from typing import Callable, Dict, List, Any
from functools import partial
from jaxopt import Bisection, FixedPointIteration


from stochastic_pgfs.laub_xia_algo import l_x_algo



def nb_degree_sequence(r0, k, N_max = 1000):
    n = k
    p = k / (r0 + k) 
    dist = nbinom(n=n, p=p)  # Scipy parameterizes with alpha / R0 + alpha
    degree_sequence = dist.pmf(range(N_max + 1))
    return degree_sequence


def nb_pgf(x, r0, k):
    """Evaluate the PGF of a Negative Binomial distribution."""
    return (1 + r0 * (1 - x)/k)**(-k)

class PGF:
    def __init__(self, poly_coef):
        """Initialize a PGF using polynomial coefficients."""
        self.exponents = jnp.arange(0, len(poly_coef))
        self.coefficients = jnp.array(poly_coef)
        
    def __call__(self, x):
        """Evaluate the PGF at x."""
        return jnp.sum(self.coefficients * jnp.power(x, self.exponents))
 
    def derivative(self, x):
        """Evaluate the derivative of the PGF at x."""
        return jnp.sum(self.exponents * self.coefficients * jnp.power(x, self.exponents - 1))


def get_root_polynomial(coef, subtract_x=True):
    """Find the smallest positive real root in (0,1) using companion matrix approach."""
    if subtract_x:
        # Subtract x to get G(x) - x = 0
        coef_adjusted = coef.at[1].add(-1.0)
    else:
        coef_adjusted = coef
        
    roots = jnp.roots(jnp.flip(coef_adjusted))
    real_roots = roots[jnp.abs(roots.imag) < 1e-8]
    unit_interval_roots = real_roots[
        (real_roots > 0.0) & 
        (jnp.isclose(real_roots, 1.0, atol=1e-4) | (real_roots < 1.0))
    ]
    
    # Handle empty array case
    if len(unit_interval_roots) > 0:
        return jnp.real(unit_interval_roots).min().item()
    else:
        return 0.5  # Default value if no valid roots found

# ====================== SCE CALCULATION METHODS ======================

def implicit_diff_non_parametric(pgf, u):
    """Compute SCE using implicit differentiation for non-parametric case."""
    powers = u ** jnp.arange(len(pgf.coefficients))
    p_prime = pgf.derivative(u)
    dr_da = -powers / p_prime
    return dr_da

def implicit_diff_parametric(G: Callable, u: float, r0: float, k: float):
    """Compute SCE using implicit differentiation for parametric case."""
    F = lambda x, r0, k: G(x, r0, k) - x  # Self consistent solution are roots of F
    dF_dr0 = jax.grad(F, argnums=1)
    dF_dk = jax.grad(F, argnums=2)
    dF_x = jax.grad(F, argnums=0)
    

    dx_dr0 = -dF_dr0(u, r0, k) / dF_x(u, r0, k)
    dx_dk = -dF_dk(u, r0, k) / dF_x(u, r0, k)
    return dx_dr0, dx_dk

def fixed_point_solver_parametric(r0, k, x0=0.5, tol=1e-10, max_iter=100):
    """Find root using FixedPointIteration solver for parametric case."""
    
    solver = FixedPointIteration(
        fixed_point_fun=nb_pgf,
        maxiter=max_iter,
        tol=tol,
        jit=True,
    )
    return  solver.run(x0, r0 = r0, k = k).params

def fixed_point_solver_non_parametric(coeffs, x0=0.5, tol=1e-10, max_iter=100):
    """Find root using FixedPointIteration solver for non-parametric case."""
    def iter_func(x,coeffs):
        pgf = PGF(coeffs)
        return pgf(x)
    
    solver = FixedPointIteration(
        fixed_point_fun=iter_func,
        maxiter=max_iter,
        tol=tol,
        jit=True,
        implicit_diff=True,  # Added this parameter to enable implicit differentiation
    )
    return solver.run(x0,coeffs).params



def analyze_sce_methods(r0_range, k_range, N_max=10, K_samples=1000, tol=1e-10):
    """
    Analyze and compare SCE calculation methods across r0 and k values.
    
    Args:
        r0_range: Range of r0 values to analyze
        k_range: Range of k values to analyze
        N_max: Maximum degree for polynomial coefficients
        K_samples: Number of samples for LX algorithm
        tol: Convergence tolerance
    
    Returns:
        pd.DataFrame: Results of the SCE analysis
    """
    results = []
    
    for r0 in r0_range:
        r0_float = float(r0)
        for k in k_range:
            k_float = float(k)
            
            # Generate degree sequence and PGF
            degree_seq = jnp.array(nb_degree_sequence(r0_float, k_float, N_max))
            pgf_coef = degree_seq.at[1].add(-1.0)  # G(x) - x
            pgf = PGF(pgf_coef)
            
            # 1. Find roots using polynomial method
            u_star_poly = get_root_polynomial(pgf_coef, subtract_x=False)
            
            # 2. Find roots using FixedPointIteration
            u_star_fp_param, fp_param_grad = jax.value_and_grad(fixed_point_solver_parametric)(r0_float, k_float)
            
            # For non-parametric case, separate value and gradient calculation
            # Use the original degree_seq for fixed point iteration, not the modified pgf_coef
            u_star_fp_nonparam = fixed_point_solver_non_parametric(degree_seq)
            fp_nonparam_grad = jax.grad(lambda coef: fixed_point_solver_non_parametric(coef))(degree_seq)
            
            # Use polynomial root for gradient calculations
            u_star = u_star_poly
            
            # METHOD 1: Implicit differentiation
            degree_seq_minus_x = degree_seq.at[1].add(-1.0)
            d_u_star_dp_nonparam = implicit_diff_non_parametric(PGF(degree_seq_minus_x), u_star)
            d_u_star_dp_param_r0, d_u_star_dp_param_k = implicit_diff_parametric(nb_pgf, u_star, r0_float, k_float)
            
            sce_abs_impl_nonparam = jnp.linalg.norm(d_u_star_dp_nonparam)
            sce_abs_impl_param_r0 = jnp.abs(d_u_star_dp_param_r0)
            sce_abs_impl_param_k = jnp.abs(d_u_star_dp_param_k)
            
            sce_rel_impl_nonparam = jnp.linalg.norm(d_u_star_dp_nonparam * degree_seq) / u_star
            sce_rel_impl_param_r0 = jnp.abs(d_u_star_dp_param_r0 * r0_float) / u_star
            sce_rel_impl_param_k = jnp.abs(d_u_star_dp_param_k * k_float) / u_star
            
            # METHOD 2: Use implicit differentiation directly with the roots found by different solvers
            sce_abs_fp_param = jnp.linalg.norm(fp_param_grad) 
            sce_abs_fp_nonparam = jnp.linalg.norm(fp_nonparam_grad)
        
            sce_rel_fp_param_r0 = jnp.abs(fp_param_grad * r0_float) / u_star
            sce_rel_fp_param_k = jnp.abs(fp_param_grad * r0_float) / u_star
            sce_rel_fp_nonparam = jnp.linalg.norm(fp_nonparam_grad * degree_seq) / u_star
            
            # METHOD 3: LaubXia algorithm
            try:
                degree_seq_numpy = np.array(degree_seq)
                sce_abs_lx = l_x_algo(
                    degree_seq_numpy,
                    K=K_samples,
                    T=1.0,
                    tol=tol,
                    max_iter=10000,
                    acceleration_method='steffensen',
                    sampling_method='orthogonal'
                )
                sce_rel_lx = sce_abs_lx * u_star
            except Exception as e:
                print(f"LaubXia failed for r0={r0_float}: {e}")
                sce_abs_lx = float('nan')
                sce_rel_lx = float('nan')
            
            # Store results
            results.append({
                'r0': r0_float,
                'k': k_float,
                'u_star_poly': float(u_star_poly),
                'u_star_fp_param': float(u_star_fp_param),
                'u_star_fp_nonparam': float(u_star_fp_nonparam),
                
                # Absolute SCE values
                'sce_abs_impl_param_r0': float(sce_abs_impl_param_r0),
                'sce_abs_impl_param_k': float(sce_abs_impl_param_k),
                'sce_abs_impl_nonparam': float(sce_abs_impl_nonparam),
                'sce_abs_fp_param': float(sce_abs_fp_param),
                'sce_abs_fp_nonparam': float(sce_abs_fp_nonparam),
                'sce_abs_lx': float(sce_abs_lx),
                
                # Relative SCE values
                'sce_rel_impl_param_r0': float(sce_rel_impl_param_r0),
                'sce_rel_impl_param_k': float(sce_rel_impl_param_k),
                'sce_rel_impl_nonparam': float(sce_rel_impl_nonparam),
                'sce_rel_fp_param_r0': float(sce_rel_fp_param_r0),
                'sce_rel_fp_param_k': float(sce_rel_fp_param_k),
                'sce_rel_fp_nonparam': float(sce_rel_fp_nonparam),
                'sce_rel_lx': float(sce_rel_lx),
            })
    
    return pd.DataFrame(results)


def main():
    """Main execution function."""

    # Define parameter values
    k_vals = jnp.linspace(0.01, 10, 1000)
    R0_vals = np.arange(0.8, 4.025, 0.025)      
    
    # Run analysis
    print("Starting SCE analysis...")
    results_df = analyze_sce_methods(R0_vals, k_vals, N_max=10, K_samples=1000)
    
    # Save results
    print('Saving results.')
    results_df.to_parquet("./test/sce_implcit_results.pq", index=False)
    print("Results saved to sce_comparison_results.pq")
    
    
    return results_df

if __name__ == "__main__":
    main()