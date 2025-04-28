import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Any
from functools import partial
from jaxopt import Bisection, FixedPointIteration
import sys
import os

# Import from stochastic_pgfs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from stochastic_pgfs.random_graphs import poisson_degree_sequence
from stochastic_pgfs.laub_xia_algo import l_x_algo

# ====================== UTILITY CLASSES AND FUNCTIONS ======================

def poisson_pgf(x, lmbd):
    """Evaluate the PGF of a Poisson distribution."""
    return jnp.exp(lmbd * (x - 1))

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

def implicit_diff_parametric(G: Callable, u: float, lmbd: float):
    """Compute SCE using implicit differentiation for parametric case."""
    F = lambda x, lmbd: G(x, lmbd) - x  # Self consistent solution are roots of F
    dF_dlmbd = jax.grad(F, argnums=1)
    dF_x = jax.grad(F, argnums=0)
    
    return -dF_dlmbd(u, lmbd) / dF_x(u, lmbd)

def fixed_point_solver_parametric(lmbd, x0=0.5, tol=1e-10, max_iter=100):
    """Find root using FixedPointIteration solver for parametric case."""
    
    solver = FixedPointIteration(
        fixed_point_fun=poisson_pgf,
        maxiter=max_iter,
        tol=tol,
        jit=True,
    )
    return  solver.run(x0, lmbd =lmbd).params

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


# Keep this function for compatibility
def bisection_root_non_parametric(coeffs):
    """Find root using bisection method for non-parametric case."""
    coef_min_x = coeffs.at[1].add(-1.0)
    pgf = PGF(coef_min_x)
    
    def func(x):
        return pgf(x) - x
    
    bisection = Bisection(optimality_fun=func, tol=1e-10, lower=0.001, upper=0.999)
    result = bisection.run()
    return result.params

# ====================== MAIN ANALYSIS FUNCTION ======================

def analyze_sce_methods(lambda_range, N_max=10, K_samples=1000, tol=1e-10):
    """
    Analyze and compare SCE calculation methods across lambda values.
    
    Args:
        lambda_range: Range of lambda values to analyze
        N_max: Maximum degree for polynomial coefficients
        K_samples: Number of samples for LX algorithm
        tol: Convergence tolerance
    
    Returns:
        pd.DataFrame: Results of the SCE analysis
    """
    results = []
    
    for lmbd in lambda_range:
        lmbd_float = float(lmbd)
        
        # Generate degree sequence and PGF
        degree_seq = jnp.array(poisson_degree_sequence(lmbd_float, N_max))
        pgf_coef = degree_seq.at[1].add(-1.0)  # G(x) - x
        pgf = PGF(pgf_coef)
        
        # 1. Find roots using polynomial method
        u_star_poly = get_root_polynomial(pgf_coef, subtract_x=False)
        
        # 2. Find roots using FixedPointIteration
        u_star_fp_param, fp_param_grad = jax.value_and_grad(fixed_point_solver_parametric)(lmbd_float)
        
        # For non-parametric case, separate value and gradient calculation
        # Use the original degree_seq for fixed point iteration, not the modified pgf_coef
        u_star_fp_nonparam = fixed_point_solver_non_parametric(degree_seq)
        fp_nonparam_grad = jax.grad(lambda coef: fixed_point_solver_non_parametric(coef))(degree_seq)
        
        # Use polynomial root for gradient calculations
        u_star = u_star_poly
        
        # METHOD 1: Implicit differentiation
        degree_seq_minus_x = degree_seq.at[1].add(-1.0)
        d_u_star_dp_nonparam = implicit_diff_non_parametric(PGF(degree_seq_minus_x), u_star)
        d_u_star_dp_param = implicit_diff_parametric(poisson_pgf, u_star, lmbd_float)
        
        sce_abs_impl_nonparam = jnp.linalg.norm(d_u_star_dp_nonparam)
        sce_abs_impl_param = jnp.abs(d_u_star_dp_param)
        
        sce_rel_impl_nonparam = jnp.linalg.norm(d_u_star_dp_nonparam * degree_seq) / u_star
        sce_rel_impl_param = jnp.abs(d_u_star_dp_param * lmbd_float) / u_star
        
        # METHOD 2: Use implicit differentiation directly with the roots found by different solvers
        sce_abs_fp_param = jnp.linalg.norm(fp_param_grad) 
        sce_abs_fp_nonparam = jnp.linalg.norm(fp_nonparam_grad)
       
        sce_rel_fp_param = jnp.abs(fp_param_grad * lmbd_float) / u_star
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
            print(f"LaubXia failed for lambda={lmbd_float}: {e}")
            sce_abs_lx = float('nan')
            sce_rel_lx = float('nan')
        
        # Store results
        results.append({
            'lambda': lmbd_float,
            'u_star_poly': float(u_star_poly),
            'u_star_fp_param': float(u_star_fp_param),
            'u_star_fp_nonparam': float(u_star_fp_nonparam),
            
            # Absolute SCE values
            'sce_abs_impl_param': float(sce_abs_impl_param),
            'sce_abs_impl_nonparam': float(sce_abs_impl_nonparam),
            'sce_abs_fp_param': float(sce_abs_fp_param),
            'sce_abs_fp_nonparam': float(sce_abs_fp_nonparam),
            'sce_abs_lx': float(sce_abs_lx),
            
            # Relative SCE values
            'sce_rel_impl_param': float(sce_rel_impl_param),
            'sce_rel_impl_nonparam': float(sce_rel_impl_nonparam),
            'sce_rel_fp_param': float(sce_rel_fp_param),
            'sce_rel_fp_nonparam': float(sce_rel_fp_nonparam),
            'sce_rel_lx': float(sce_rel_lx),
        })
    
    return pd.DataFrame(results)

# ====================== PLOTTING FUNCTIONS ======================

def plot_results(results_df, figsize=(18, 15)):
    """Create plots comparing the different SCE calculation methods."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Compare root finding methods
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results_df['lambda'], 1 - results_df['u_star_poly'], 'o-', label='Polynomial')
    ax1.plot(results_df['lambda'], 1 - results_df['u_star_fp_param'], 'x-', label='FixedPoint Param')
    ax1.set_xlabel(r'Mean Degree ($\lambda$)')
    ax1.set_ylabel(r'Extinction Probability ($1-u^*$)')
    ax1.set_title('Comparison of Root Finding Methods')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Compare absolute SCE values
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results_df['lambda'], results_df['sce_abs_impl_param'], 'o-', label='Implicit Param')
    ax2.plot(results_df['lambda'], results_df['sce_abs_fp_param'], '^-', label='FixedPoint Param')
    ax2.plot(results_df['lambda'], results_df['sce_abs_lx'], 's-', label='LaubXia')
    ax2.set_xlabel(r'Mean Degree ($\lambda$)')
    ax2.set_ylabel('Absolute SCE')
    ax2.set_title('Comparison of Absolute Parametric SCE Values')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Compare relative SCE values
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.plot(results_df['lambda'], results_df['sce_abs_fp_nonparam'], 'd-', label='FixedPoint NonParam')  # Updated label
    ax3.plot(results_df['lambda'], results_df['sce_abs_impl_nonparam'], 'x-', label='Implicit Non-param')
    ax3.set_xlabel(r'Mean Degree ($\lambda$)')
    ax3.set_ylabel('Absolute SCE Non-Parametric')
    ax3.set_title('Comparison of Absolute Non-Parametric SCE Values')
    ax3.legend()
   
    # Plot  3
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(results_df['lambda'], results_df['sce_rel_impl_param'], 'o-', label='Implicit Param')
    ax4.plot(results_df['lambda'], results_df['sce_rel_fp_param'], '^-', label='FixedPoint Param')
    ax4.plot(results_df['lambda'], results_df['sce_rel_lx'], 's-', label='LaubXia')
    ax4.set_xlabel(r'Mean Degree ($\lambda$)')
    ax4.set_ylabel('Relative SCE Parametric')
    ax4.set_title('Comparison of Relative SCE Values')
    ax4.legend()
    ax4.grid(True)
    
    
    ax5 = fig.add_subplot(gs[2, 1])
        
    ax5.plot(results_df['lambda'], results_df['sce_rel_fp_nonparam'], 'd-', label='FixedPoint NonParam')  # Updated label
    ax5.plot(results_df['lambda'], results_df['sce_rel_impl_nonparam'], 'x-', label='Implicit Non-param')
    
    ax5.set_xlabel(r'Mean Degree ($\lambda$)')
    ax5.set_ylabel('Relative SCE Non-parametric')
    ax5.set_title('Comparison of Relative SCE Values')
    ax5.legend()
    ax5.grid(True)
    

    
    # Add a title plot
    ax_title = fig.add_subplot(gs[0, 1])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'SCE Calculation Methods Comparison',
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ====================== MAIN EXECUTION ======================

def main():
    """Main execution function."""
    # Define lambda range
    lambda_range = jnp.linspace(0.1, 2.0, 50)
    
    # Run analysis
    print("Starting SCE analysis...")
    results_df = analyze_sce_methods(lambda_range, N_max=10, K_samples=1000)
    
    # Save results
    results_df.to_csv("sce_comparison_results.csv", index=False)
    print("Results saved to sce_comparison_results.csv")
    
    # Plot results
    print("Generating plots...")
    fig = plot_results(results_df)
    plt.savefig("sce_comparison_results.png", dpi=300)
    plt.show()
    
    return results_df

if __name__ == "__main__":
    main()




