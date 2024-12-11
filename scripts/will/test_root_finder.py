from stochastic_pgfs.laub_xia_algo import iterate_until_convergence, G1,get_outbreak_size,l_x_algo
from stochastic_pgfs.random_graphs import poisson_degree_sequence, powerlaw_degree_sequence
import numpy as np
from numba import config
import logging
import timeit

# Enable Numba logging
#config.LOGNAME = "numba.jitclass"
config.DEBUG = True
logging.basicConfig(level=logging.INFO)

def test_root_finder():
    # Create a simple degree sequence (e.g., Poisson with mean 2)
    N_max = 10
    my_poly_coef = poisson_degree_sequence(2.0)
    degree_sequence = np.vstack((np.arange(0, my_poly_coef.shape[0], 1), my_poly_coef)).T
    
    # Stack degree and probability into required format
    
    print("Testing root finder with degree sequence:")
    print(f"Degree sequence shape: {degree_sequence.shape}")
    print(f"First few entries (k, pk):\n{degree_sequence[:5]}")
    
    # Test at different T values
    T_values = [0.3, 0.50000001, 0.7, 1.0]
    
    for T in T_values:
        print(f"\nTesting T = {T}")
        
        # Time the root finder
        start_time = timeit.default_timer()
        u1, u2 = iterate_until_convergence(degree_sequence, T=T, tol=1e-10, max_iter=1e11)
        elapsed = timeit.default_timer() - start_time
        
        print(f"Time taken: {elapsed:.6f} seconds")
        
        # Time the lx_algo
        start_time = timeit.default_timer()
        stability_measure = l_x_algo(my_poly_coef, T=T, K=1000, delta=0.001)
        elapsed = timeit.default_timer() - start_time
        
        print(f"lx_algo Time taken: {elapsed:.6f} seconds")
        print(f"Stability measure: {stability_measure:.10f}")
        
        get_outbreak_size(my_poly_coef, T)
        
        # Verify the solution
        final_residual = abs(G1(u2, degree_sequence, T) - u2)
        print(f"Found root: {u2:.10f}")
        print(f"Residual: {final_residual:.2e}")
        print(f"Outbreak size: {1-u2:.10f}")

if __name__ == "__main__":
    print("Testing Numba compilation...")
    test_root_finder()
