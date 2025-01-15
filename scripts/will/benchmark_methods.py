import numpy as np
import time
from scipy.stats import poisson
import matplotlib.pyplot as plt
from stochastic_pgfs.laub_xia_algo import iterate_with_acceleration

def create_poisson_degree_sequence(lambda_param, max_degree=50):
    """Create a truncated Poisson degree sequence"""
    k = np.arange(0, max_degree)
    pk = poisson.pmf(k, lambda_param)
    # Normalize to ensure sum = 1 after truncation
    pk = pk / pk.sum()
    return pk

def benchmark_methods(pk, T=1.0, n_trials=100):
    """Benchmark different iteration methods"""
    methods = ['naive', 'aitken', 'steffensen']
    times = {method: [] for method in methods}
    results = {method: [] for method in methods}
    
    for method in methods:
        for _ in range(n_trials):
            start_time = time.time()
            u1, u2 = iterate_with_acceleration(
                np.vstack((np.arange(len(pk)), pk)).T,
                T=T,
                method=method
            )
            end_time = time.time()
            times[method].append(end_time - start_time)
            results[method].append(u2)
    
    return times, results

def plot_results(times, results):
    """Plot timing results and convergence values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot timing boxplots
    ax1.boxplot([times[method] for method in times.keys()], labels=times.keys())
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Computation Time by Method')
    
    # Plot convergence values
    for method in results.keys():
        ax2.hist(results[method], alpha=0.5, label=method, bins=20)
    ax2.set_xlabel('Converged Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Converged Values')
    ax2.legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create a Poisson degree sequence
    lambda_param = 2.5  # Mean degree
    pk = create_poisson_degree_sequence(lambda_param)
    
    # Run benchmarks
    print("Running benchmarks...")
    times, results = benchmark_methods(pk)
    
    # Print summary statistics
    print("\nTiming Results (seconds):")
    for method in times:
        mean_time = np.mean(times[method])
        std_time = np.std(times[method])
        mean_result = np.mean(results[method])
        std_result = np.std(results[method])
        print(f"\n{method.capitalize()} Method:")
        print(f"Average time: {mean_time:.6f} ± {std_time:.6f}")
        print(f"Average result: {mean_result:.6f} ± {std_result:.6f}")
    
    # Plot results
    fig = plot_results(times, results)
    plt.show()
