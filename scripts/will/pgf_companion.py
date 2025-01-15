import torch
import numpy as np
from scipy.stats import poisson

def generate_poisson_pgf(lambda_param, max_degree=50):
    """Generate normalized Poisson degree sequence"""
    k = np.arange(max_degree)
    pk = poisson.pmf(k, lambda_param)
    pk = pk / pk.sum()  # Normalize
    return torch.tensor(pk, dtype=torch.float64)

def G_1_minus_u_coeffs(pk):
    """
    Compute coefficients for G_1(u) - u function
    pk: degree probabilities [p0, p1, p2, ...]
    """
    n = len(pk)
    degrees = torch.arange(n, dtype=torch.float64)
    
    # Compute coefficients for G_1(u)
    G1_coeffs = pk * degrees / degrees.dot(pk)
    # Subtract u (add -1 to linear term)
    G1_coeffs[1] -= 1
    return G1_coeffs

def find_roots(coeffs):
    """Find roots using companion matrix method"""
    # Normalize coefficients by leading coefficient
    leading_coeff = coeffs[-1]
    if abs(leading_coeff) < 1e-10:
        leading_coeff = torch.sign(leading_coeff) * 1e-10
    normalized_coeffs = coeffs / leading_coeff
    
    # Create companion matrix
    n = len(coeffs)
    C = torch.zeros((n-1, n-1), dtype=torch.float64)
    
    # Set subdiagonal to 1
    if n > 2:
        idx = torch.arange(n-2)
        C[idx+1, idx] = 1.0
    
    # Set last row to negative normalized coefficients
    C[-1] = -normalized_coeffs[:-1] / normalized_coeffs[-1]
    # Find eigenvalues
    roots = torch.linalg.eigvals(C)
    # Filter out numerical noise in imaginary parts
    roots = torch.where(
        torch.abs(roots.imag) < 1e-10,
        roots.real,
        roots
    )
    
    # Sort roots by magnitude
    roots = roots[torch.argsort(torch.abs(roots))]
    
    return roots

if __name__ == "__main__":
    # Generate Poisson degree sequence
    lambda_param = 2.5  # Mean degree
    pk = generate_poisson_pgf(lambda_param)
    
    # Compute G_1(u) - u coefficients
    g1_minus_u_coeffs = G_1_minus_u_coeffs(pk)
    
    # Find roots
    roots = find_roots(g1_minus_u_coeffs)
    
    print(f"Poisson parameter λ = {lambda_param}")
    print("\nRoots of G_1(u) - u = 0:")
    for i, root in enumerate(roots):
        if abs(root.imag) < 1e-10:
            print(f"Root {i+1}: {root.real:.8f}")
