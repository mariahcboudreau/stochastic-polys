import numpy as np
from stochastic_pgfs.laub_xia_algo import _steffensen_iteration,_aitken_iteration, iterate_with_acceleration, G1, l_x_algo,get_outbreak_size
from scipy.stats import poisson
import matplotlib.pyplot as plt
import time
import tracemalloc
tracemalloc.start()
import logging

def create_poisson_degree_sequence(lambda_param, max_degree=50):
    """Create a truncated Poisson degree sequence"""
    k = np.arange(0, max_degree)
    pk = poisson.pmf(k, lambda_param)
    # Normalize to ensure sum = 1 after truncation
    pk = pk / pk.sum()
    return pk

def calculate_outbreak_size_and_time(lambda_param, T_vals, method, max_degree=50, tol=1e-5, max_iter=100, n_trials=100):
    # Create a Poisson degree sequence
    pk = create_poisson_degree_sequence(lambda_param, max_degree)
    pk = np.vstack((np.arange(len(pk)), pk)).T
    
    outbreak_sizes = []
    run_times = []
    
    for T in T_vals:
        total_outbreak_size = 0
        total_time = 0
    
        for _ in range(n_trials):
            # Define the function to iterate
            def G1_func(u):
                return G1(u, pk, T)
            
            # Initial guess
            u0 = 0.5
            
            # Measure time
            start_time = time.time()
            # Run the specified iteration method
            if method == 'steffensen':
                u_star = iterate_with_acceleration(G1_func, u0, tol=tol, max_iter=max_iter)
            elif method == 'aitken':
                u_star, _ = iterate_with_acceleration(pk, T=T, method='aitken', tol=tol, max_iter=max_iter)
            elif method == 'naive':
                u_star, _ = iterate_with_acceleration(pk, T=T, method='naive', tol=tol, max_iter=max_iter)
            else:
                raise ValueError("Method must be 'steffensen', 'aitken', or 'naive'")
            
            end_time = time.time()
            
            # Calculate outbreak size
            outbreak_size = 1 - u_star
            total_outbreak_size += outbreak_size
            total_time += (end_time - start_time)
        
        average_outbreak_size = total_outbreak_size / n_trials
        average_time = total_time / n_trials
        
        outbreak_sizes.append(average_outbreak_size)
        run_times.append(average_time)
    
    return outbreak_sizes, run_times

def calculate_outbreak_size_time_and_sce(pk, T, method='steffensen', **kwargs):
    """Calculate outbreak size, computation time, and SCE with proper memory management"""
    try:
        # Create arrays with proper memory alignment
        outbreak_size = np.zeros(1, dtype=np.float64)
        computation_time = np.zeros(1, dtype=np.float64)
        
        # Time the computation
        start_time = time.perf_counter()
        
        # Use context manager for safer array operations
        with np.errstate(all='raise'):
            # Calculate outbreak size
            outbreak_size[0] = get_outbreak_size(pk, T)
            
            # Calculate SCE value
            sce = l_x_algo(
                pk,
                T=T,
                max_iter=kwargs.get('max_iter', 1000),
                tol=kwargs.get('tol', 1e-6),
                acceleration_method=method,
                K=kwargs.get('K', 100),
                sampling_method=kwargs.get('sampling_method', 'random')
            )
        
        computation_time[0] = time.perf_counter() - start_time
        
        return outbreak_size[0], computation_time[0], sce
        
    except Exception as e:
        logging.error(f"Error in calculation: {str(e)}")
        return np.nan, np.nan, np.nan
    finally:
        # Ensure memory is freed
        outbreak_size = None
        computation_time = None

def plot_results(T_vals, results, ylabel, title):
    plt.figure(figsize=(10, 6))
    for method, values in results.items():
        plt.plot(T_vals, values, marker='o', linestyle='-', label=method)
    plt.xlabel('Transmission Rate (T)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def test_steffensen():
    lambda_param = 2.5  # Mean degree
    T_vals = np.linspace(0.01, 1, 50)  # Transmission rate values
    max_degree = 100
    tol = 1e-10
    max_iter = int(1e10)
    n_trials = 1
    K = 100
    
    methods = ['steffensen', 'aitken', 'naive']
    #methods = ['steffensen', 'naive']
    #methods = ['naive']
    outbreak_size_results = {}
    iteration_time_results = {}
    sce_results = {}
    
    # Create degree sequence once
    pk = create_poisson_degree_sequence(lambda_param, max_degree)
    
    for method in methods:
        print(f"Testing {method} method...")
        outbreak_sizes = []
        iteration_times = []
        sces = []
        
        for T in T_vals:
            print("T:", T)
            # Call with proper kwargs
            outbreak_size, computation_time, sce = calculate_outbreak_size_time_and_sce(
                pk=pk,
                T=T,
                method=method,
                max_iter=max_iter,
                tol=tol,
                K=K
            )
            
            outbreak_sizes.append(outbreak_size)
            iteration_times.append(computation_time)
            sces.append(sce)
        
        outbreak_size_results[method] = outbreak_sizes
        iteration_time_results[method] = iteration_times
        sce_results[method] = sces
   
    print(sce_results) 
    # Plot the results
    plot_results(T_vals, outbreak_size_results, 'Average Outbreak Size', 'Average Outbreak Size vs Transmission Rate')
    plot_results(T_vals, iteration_time_results, 'Average Computation Time (seconds)', 'Computation Time vs Transmission Rate')
    plot_results(T_vals, sce_results, 'SCE Value', 'SCE vs Transmission Rate')

def analyze_sce_variance(lambda_param, T, method, K_vals, max_degree=50, tol=1e-5, max_iter=100, n_trials=10):
    """
    Analyze how SCE variance changes with K for different sampling methods
    """
    pk = create_poisson_degree_sequence(lambda_param, max_degree)
    
    random_variances = []
    orthogonal_variances = []
    
    for K in K_vals:
        print(f"Testing K={K}")
        random_sces = []
        orthogonal_sces = []
        
        for _ in range(n_trials):
            # Calculate SCE with random sampling
            try:
                random_sce = l_x_algo(
                    pk,
                    T=T,
                    max_iter=max_iter,
                    tol=tol,
                    acceleration_method=method,
                    K=K,
                    sampling_method='random'
                )
                random_sces.append(random_sce)
            except Exception as e:
                print(f"Random sampling failed: {str(e)}")
                random_sces.append(np.nan)
                
            # Calculate SCE with orthogonal sampling
            try:
                orthogonal_sce = l_x_algo(
                    pk,
                    T=T,
                    max_iter=max_iter,
                    tol=tol,
                    acceleration_method=method,
                    K=K,
                    sampling_method='orthogonal'
                )
                orthogonal_sces.append(orthogonal_sce)
            except Exception as e:
                print(f"Orthogonal sampling failed: {str(e)}")
                orthogonal_sces.append(np.nan)
       
        # Calculate variances
        random_variances.append(np.nanvar(random_sces))
        orthogonal_variances.append(np.nanvar(orthogonal_sces))
    
    return random_variances, orthogonal_variances

    
    
def plot_variance_comparison(K_vals, random_variances, orthogonal_variances, title):
    plt.figure(figsize=(10, 6))
    try:
        plt.plot(K_vals, random_variances, marker='o', linestyle='-', label='Random Sampling')
        plt.plot(K_vals, orthogonal_variances, marker='s', linestyle='-', label='Orthogonal Sampling')
        plt.xlabel('Number of Samples (K)')
        plt.ylabel('SCE Variance')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
    finally:
        plt.close('all')  # Ensure all figures are cleaned up 
    
def test_sampling_methods():
    # Parameters
    lambda_param = np.float64(2.0)  # Ensure consistent types
    T = np.float64(0.5)
    max_degree = int(1000)
    tol = np.float64(1e-10)
    max_iter = int(1e8)
    n_trials = int(10)
    
    # Create K values safely
    K_vals = np.ascontiguousarray(
        np.logspace(1, 5, 10, dtype=np.int32)
    )
    
    
    methods = ['steffensen']
    for method in methods:
        print(f"\nTesting {method} method...")
        try:
            # Create new arrays for each iteration
            random_vars = np.zeros_like(K_vals, dtype=np.float64)
            ortho_vars = np.zeros_like(K_vals, dtype=np.float64)
            
            random_variances, orthogonal_variances = analyze_sce_variance(
                lambda_param=lambda_param,
                T=T,
                method=method,
                K_vals=K_vals.copy(),
                max_degree=max_degree,
                tol=tol,
                max_iter=max_iter,
                n_trials=n_trials
            )
            
            # Copy results to prevent memory issues
            random_vars[:] = np.array(random_variances, dtype=np.float64)
            ortho_vars[:] = np.array(orthogonal_variances, dtype=np.float64)
            
            # Plot with copied arrays
            plot_variance_comparison(
                K_vals=K_vals.copy(),
                random_variances=random_vars,
                orthogonal_variances=ortho_vars,
                title=f'SCE Variance vs K for {method.capitalize()} Method'
            )
        except Exception as e:
            print(f"Error in {method} method: {e}")
            continue
        finally:
            # Clear any remaining plots
            plt.close('all')


test_steffensen()
test_sampling_methods()

