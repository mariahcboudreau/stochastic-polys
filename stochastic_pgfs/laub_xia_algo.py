import numpy as np
from numpy.linalg import norm, eig
from scipy.linalg import schur
from scipy.special import factorial2
import matplotlib.pyplot as plt
import random
import time
from numba import jit
import tqdm
import sys
import os

import os
import sys
sys.path.insert(1,os.getcwd())
from stochastic_pgfs.pgfs import *
import logging


# Approximation for the Wallis factor
def omega(n):
    """
    Approximation of the Wallis factor for a given input
    Eq. 101 in https://epubs.siam.org/doi/10.1137/0915003
    Parameters:
    n (float): The input value.
    Returns:
    float: The calculated value of the wallis factor
    """
    a1 = np.sqrt(2 / (np.pi * (n - 0.5)))
    a2 = np.sqrt((184 * n**4)/ (1 + 23 * n + 23 * n**2 + 184 * n**4))
    return a1 * a2

def generate_sphere_point(dim):
    """
    Generates a random point on the surface of a unit sphere in the given dimension.

    Parameters:
    dim (int): The dimension of the sphere.

    Returns:
    numpy.ndarray: A random point on the surface of the unit sphere.
    """
    vec = np.random.randn(dim)
    return vec / norm(vec)


def derivative(p_kCoeff):
    """
    Compute the derivative of a polynomial represented by its coefficients.

    Parameters:
    p_kCoeff (array-like): Coefficients of the polynomial.

    Returns:
    array-like: Coefficients of the derivative polynomial.

    """
    p_kCoeff = np.array(p_kCoeff)
    
    if p_kCoeff.size == 0:
        return np.array([0])
    elif p_kCoeff.size == 1:
        return np.array([0])
    else:
        primeCoeff = np.empty((len(p_kCoeff)-1))
        for p in range(len(p_kCoeff)-1):
            primeCoeff[p] = p_kCoeff[p+1]*(p+1)
        
    return primeCoeff




def _perturb_polynomial(poly_coefs, delta, alpha_i, perturbation_type):
    """
    Perturbs the coefficients of a polynomial based on the given perturbation type, step 4a from algorithm 1 in DOI:10.1137/070702242

    Args:
        poly_coefs (list): The coefficients of the polynomial.
        delta (float): The perturbation factor.
        alpha_i (float): The alpha value.
        perturbation_type (str): The type of perturbation. Must be either "additive" or "multiplicative".

    Returns:
        list: The perturbed coefficients of the polynomial.

    Raises:
        ValueError: If the perturbation type is neither "additive" nor "multiplicative".
    """
    # Perform perturbation, either additive or multiplicative
    perturbed_coefs = np.copy(poly_coefs)
    if perturbation_type == "additive":
        perturbed_coefs[:,1] = poly_coefs[:,1] * (1 + delta * alpha_i)
    # elif perturbation_type == "multiplicative": 
    #     perturbed_coefs = poly_coefs * (1 * delta * alpha_i)
    else:
        raise ValueError("Perturbation type must be either additive or multiplicative")
   
    #normalize the coefficients
    perturbed_coefs[:,1] = perturbed_coefs[:,1]/sum(perturbed_coefs[:,1]) 
    return perturbed_coefs



def l_x_metric(og_roots, perturbed_roots, delta, K, N):
    """
    Calculate the Laub-Xia stability metric as defined in the paper.
    
    Args:
        og_roots: Original unperturbed roots
        perturbed_roots: Perturbed roots
        delta: Perturbation size
        K: Number of perturbation vectors
        N: Dimension of the system
    
    Returns:
        float: Computed SCE value
    """
    # Calculate relative error
    relative_error = np.abs(perturbed_roots - og_roots) / (delta * np.abs(og_roots))
    # Apply normalization factors
    return omega(K) / omega(N) * np.linalg.norm(relative_error)


@jit(nopython=True)
def G0(x, pk, T):
    x = (1 - T) + T * x
    return np.power(x, pk[:, 0]).dot(pk[:, 1])

@jit(nopython=True)
def G1(x, pk, T):
    """Calculate G1 with aligned arrays"""
    x = np.float64(1 - T) + np.float64(T) * x
    # Split array access and ensure float64
    pk_0 = pk[:, 0].astype(np.float64)
    pk_1 = pk[:, 1].astype(np.float64)
    
    numerator = np.power(x, pk_0 - 1) * pk_1 * pk_0
    denominator = pk_0.dot(pk_1)  # Use dot product on contiguous arrays
    return numerator.sum() / denominator

def G_1_minus_u(x, pk, T):
    """
    Compute G_1 - u, which subtracts 1 from the coefficient for the linear term.
    """
    pk = np.copy(pk)
    pk_G_1=  G1(x, pk, T)
    pk_G_1[1,1] -= 1
    return pk_G_1
    

@jit(nopython=True)
def G1_prime(x, pk, T):
    x = (1 - T) + T * x
    numerator = (pk[:, 0] - 1) * pk[:, 0] * pk[:, 1] * np.power(x, pk[:, 0] - 2)
    #denominator = (pk[:, 0] - 1) * pk[:, 0] * pk[:, 1]
    denominator =  pk[:, 0] * pk[:, 1]
    return numerator.sum() / denominator.sum()

def get_outbreak_size(my_degree_sequence, T, method='steffensen'):
    """
    Calculate outbreak size using specified acceleration method
    
    Args:
        my_degree_sequence: Degree sequence
        T: Transmission rate
        method: Acceleration method ('steffensen', 'aitken', or 'naive')
        
    Returns:
        float: Outbreak size (1-u2)
    """
    pk = np.vstack((np.arange(0, my_degree_sequence.shape[0], 1), my_degree_sequence)).T
    u1, u2 = iterate_with_acceleration(
        pk,
        T=T, 
        method=method,
        tol=1e-5, 
        max_iter=int(1e5)
    )
    outbreak_size = 1 - u2
    return outbreak_size

    

@jit(nopython=True)
def _aitken_iteration(pk, T, tol, max_iter, acceleration_interval, usol):
    """
    Perform iteration using Aitken's acceleration method with safe array handling.
    """
    # Ensure input arrays are properly typed
    pk = pk.astype(np.float64)
    T = np.float64(T)
    tol = np.float64(tol)
    usol = np.float64(usol)
    
    # Pre-allocate fixed-size array for sequence values
    u_values = np.zeros(3, dtype=np.float64)  # Only need last 3 values
    
    u1 = usol
    u2 = G1(u1, pk, T)
    
    # Initialize sequence
    u_values[0] = u1
    u_values[1] = u2
    u_values[2] = G1(u2, pk, T)
    
    for i in range(max_iter):
        if abs(u2 - u1) < tol:
            break
            
        if i % acceleration_interval == 0 and i > 0:
            # Apply Aitken acceleration using most recent values
            denominator = u_values[2] - 2.0*u_values[1] + u_values[0]
            if abs(denominator) > tol:
                # Safe acceleration computation
                u2 = u_values[0] - ((u_values[1] - u_values[0])**2) / denominator
                
                # Reset sequence with accelerated value
                u_values[0] = u2
                u_values[1] = G1(u2, pk, T)
                u_values[2] = G1(u_values[1], pk, T)
            else:
                # Regular iteration if denominator is too small
                u1 = u2
                u2 = G1(u1, pk, T)
                
                # Shift values
                u_values[0] = u_values[1]
                u_values[1] = u_values[2]
                u_values[2] = u2
        else:
            # Regular iteration
            u1 = u2
            u2 = G1(u1, pk, T)
            
            # Shift values
            u_values[0] = u_values[1]
            u_values[1] = u_values[2]
            u_values[2] = u2
    
    return u1, u2

@jit(nopython=True)
def _naive_iteration(pk, T, tol, max_iter, usol):
    """Perform simple fixed-point iteration."""
    pk = pk.astype(np.float64)
    u1 = np.float64(usol)
    u2 = G1(u1, pk, T)
    
    for i in range(max_iter):
        if abs(u2 - u1) < tol:
            break
        u1 = u2
        u2 = G1(u1, pk, T)
    return u1, u2

@jit(nopython=True)
def _steffensen_iteration(pk, T, tol, max_iter, usol):
    """Perform Steffensen iteration."""
    pk = pk.astype(np.float64)
    u1 = np.float64(usol)
    u2 = u1
    
    for i in range(max_iter):
        p0 = u1
        p1 = G1(p0, pk, T)
        p2 = G1(p1, pk, T)
        
        denominator = p2 - 2*p1 + p0
        if abs(denominator) < tol:
            u2 = p0
            break
            
        u2 = p0 - (p1 - p0)**2 / denominator
        
        if abs(u2 - u1) < tol:
            break
            
        u1 = u2
    return u1, u2

@jit(nopython=True)
def iterate_with_acceleration(pk, T=1, method='aitken', tol=1e-5, usol=0.5, max_iter=int(1e4), acceleration_interval=10):
    """Iterate with acceleration and aligned arrays."""
    pk = pk.astype(np.float64)
    T = np.float64(T)
    tol = np.float64(tol)
    
    if method not in ['aitken', 'steffensen', 'naive']:
        raise ValueError("Method must be 'aitken', 'steffensen', or 'naive'")
   
    if method == 'naive':
        return _naive_iteration(pk, T, tol, max_iter, usol)
    elif method == 'aitken':
        return _aitken_iteration(pk, T, tol, max_iter, acceleration_interval, usol)
    elif method == 'steffensen':
        return _steffensen_iteration(pk, T, tol, max_iter, usol)

def l_x_algo(
    my_poly_coef,
    K=1000,
    conditions=None,
    delta=0.001,
    T=1.0,
    perturbation_type="additive",
    bifurcation=False,
    derivative_test=True,
    max_iter=int(1e8),
    tol=1e-10,
    acceleration_method='steffensen',
    sampling_method='orthogonal',
    assume_G1_equal = True
):
    """Calculate stability measure using the Laub-Xia algorithm"""
    # Ensure input arrays are contiguous and properly typed
    T = np.float64(T)

    if assume_G1_equal:
        my_poly_coef= my_poly_coef/np.arange(1,len(my_poly_coef)+1)
        my_poly_coef = np.insert(my_poly_coef, 0, 0)

    my_poly_coef = np.ascontiguousarray(my_poly_coef)
    my_poly_coef = np.vstack((np.arange(0, my_poly_coef.shape[0], 1), my_poly_coef)).T.copy()
    
    # print(my_poly_coef)

    if conditions is None:
        conditions = []
    
    N = my_poly_coef.shape[0]
    
    # Generate perturbation vectors
    if sampling_method == 'orthogonal':
        # Generate matrix of random vectors and perform QR decomposition
        Z = np.random.randn(N, K)
        Q, R = np.linalg.qr(Z, mode='reduced')
        Z = np.ascontiguousarray(Q[:, :K])  # Ensure we only take K columns
        K = Z.shape[1]  # Update K to match actual number of vectors
    else:
        # Generate random unit vectors
        Z = np.empty((N, K), dtype=np.float64)
        for i in range(K):
            vec = np.random.randn(N)
            Z[:, i] = vec / np.linalg.norm(vec)
    
    # Pre-allocate arrays for results
    sce_values = np.zeros(K, dtype=np.float64)
    
    og_roots, _ = iterate_with_acceleration(
        my_poly_coef, 
        T=T, 
        method=acceleration_method,
        tol=tol, 
        max_iter=max_iter
    )
    
    
    # delta = np.float64(2**(-16))
    delta = np.sqrt(np.abs(og_roots) * np.finfo(np.float64).eps)
    
    # Process perturbations
    og_roots_list = np.zeros((K), dtype=np.float64)
    perturbed_roots_list = np.zeros((K), dtype=np.float64)
    SCE_list = []
    
    # Calculate function value at fixed point
    function_value = G1(og_roots, my_poly_coef, T)
    
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
            max_iter=max_iter,
            usol=og_roots
        )
       
        perturbed_roots_list[i] = perturbed_roots
   


    # # Debugging: Print the values of perturbed_roots_list and og_roots
    # print("perturbed_roots_list:", perturbed_roots_list)
    # print("og_roots:", og_roots)
    # print("function_value:", function_value)    


    # Calculate scaled gradient norm
    gradient_norm = np.linalg.norm(np.abs(perturbed_roots_list - og_roots) / delta * np.abs(og_roots))
    # print(N)
    # Scale by function value and Wallis factors
    #return gradient_norm
    #return (gradient_norm) *omega(K) / omega(N)
    # print("gradient_norm:", gradient_norm)

    return (gradient_norm / np.abs(function_value)) * omega(K) / omega(N)
