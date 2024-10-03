import numpy as np
from numpy.linalg import norm, eig
from scipy.linalg import schur
from scipy.special import factorial2
import matplotlib.pyplot as plt
import random
import time

from stochastic_pgfs.pgfs import *


def double_factorial(n):
    """
    Calculate the double factorial of a number.

    Parameters:
    n (int): The number for which to calculate the double factorial.

    Returns:
    int: The double factorial of the given number.

    Example:
    >>> double_factorial(5)
    15
    """
    return factorial2(n - 1)


# Approximation for the Wallis factor
def omega(n):
    """
    Approximation of the Wallis factor for a given input
    Eq. 101 in https://epubs.siam.org/doi/10.1137/0915003
    Parameters:
    n (float): The input value.
    Returns:
    float: The calculated value of the wallace factor
    """
    a1 = np.sqrt(2 / (np.pi * (n - 0.5)))
    a2 = np.sqrt((184 * n**4) + 23 * n + 23 * n**2 + 184 * n**4)
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


def companion_matrix(coefs):
    """
    Constructs the companion matrix for a given list of coefficients.

    Args:
        coefs (list or array-like): List of coefficients of a polynomial.

    Returns:
        numpy.ndarray: The companion matrix.

    Raises:
        ValueError: If the input has less than 2 coefficients.
    """
    coefs = np.array(coefs)
    n = len(coefs)
    if n < 2:
        raise ValueError("The input must have at least 2 coefficients.")
    A = np.zeros((n - 1, n - 1))
    if n > 2:
        A[1:, :-1] = np.eye(n - 2)
    if coefs[0] != 0:
        A[:, -1] = -np.flip(coefs[1:]) / coefs[0]  # Divide by the coefficient of the highest order term
    else:
        A[:, -1] = -np.flip(coefs[1:])
    return A


# def polynomial_roots(poly_coef):
#     """
#     Calculate the roots of a polynomial.

#     Parameters:
#     poly_coef (array-like): Coefficients of the polynomial in descending order.

#     Returns:
#     array-like: Array of complex numbers representing the roots of the polynomial.
#     """
#     C = companion_matrix(poly_coef)
#     return np.linalg.eigvals(C)




def fast_polynomial_roots(poly_coef, solve_root = True,tol = 1e-5):
    """
    Calculate the roots of a polynomial.

    Parameters:
    poly_coef (array-like): Coefficients of the polynomial in descending order.
    tol(float): the tolerance at which the iterative root returns a solution

    Returns:
    array-like: Array of complex numbers representing the roots of the polynomial.
    """
   
    p = np.flip(poly_coef)
    #breakpoint()
    #p = poly_coef
    
    u1 = 0.5
    u2 = np.polyval(p, u1)
    while  abs(u2-u1) > 10**(-5):
        u1 = u2.copy()
        u2 = np.polyval(p,u1)
    usol = np.array([u2])
    outbreaksol = np.array([1-u2])
    
    if solve_root:
        return usol
    else:
        return outbreaksol


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



def in_bounds(coefs):
    """
    Check if the coefficients of a polynomial are within a specified range.

    Args:
        coefs (numpy.ndarray): The coefficients of the polynomial.
    Returns:
        numpy.ndarray: A boolean array indicating whether each coefficient is within the specified range.
    """
    epsilon = np.finfo(float).eps
    epsilon = 1e-6
    lower_bound = -epsilon
    upper_bound = 1.0 - epsilon
    # real_coefs = np.real(coefs)
    # Check if real_coefs are greater than or equal to lower_bound
    lower_check = np.logical_or(coefs >= lower_bound, np.isclose(coefs, lower_bound, atol=epsilon))
    # Check if real_coefs are less than upper_bound
    upper_check = np.logical_or(coefs <= upper_bound, np.isclose(coefs, upper_bound, atol=epsilon, rtol=0))
    
    return np.logical_and(lower_check, upper_check)


def is_real(coefs):
    """
    Check if the given coefficients are real numbers.

    Parameters:
    coefs (array-like): The coefficients to be checked.

    Returns:
    bool: True if all coefficients are real, False otherwise.
    """
    return np.isclose(np.imag(coefs), 0)


def _filter_roots(poly_roots, conditions):
    """
    Filters the polynomial roots based on the specified conditions.

    Args:
        poly_roots (ndarray): Array of polynomial roots.
        conditions (list): List of conditions to filter the roots, should be is_real and in_bounds

    Returns:
        ndarray: Filtered array of polynomial roots.

    Raises:
        ValueError: If no roots meet the conditions.

    """
    if conditions:  #if conditions are specified filter roots by conditions to ensure they are real and within bounds
        all_conditions = np.logical_and.reduce(
            [cond(poly_roots) for cond in conditions]
        )
        if len(poly_roots[all_conditions]) >= 1:
                poly_roots = poly_roots[all_conditions]
        else:
            # raise ValueError("No roots meet the conditions")
            poly_roots = np.array([1])

    return poly_roots

def _solve_self_consistent_equation(degree_sequence, conditions=[is_real, in_bounds],derivative_test = True, solve_root = True):
        """
        Solve the self-consistent equation for a PGF G(u) = u by finding the roots of G(u) - u = 0.
        See sections C and D from DOI: 10.1103/PhysRevE.64.026118
        The roots will correspond to 1 - the giant component size.
        Parameters:
        - degree_sequence (list): The degree sequence of the graph.
        - conditions (list, optional): A list of conditions that the roots must satisfy.
            Defaults to [is_real, in_bounds].
        
        Returns:
        - filtered_roots (list): The filtered roots that satisfy the given conditions.
       """
       
        if (sum(derivative(degree_sequence)) < 1 and derivative_test == True): 
            if solve_root:
                return np.array([1])
            else:
                return np.array([0])
            
        else: 
            filtered_roots = np.array([1])
            my_pgf_coef = make_G_u_minus_u(degree_sequence)  # coefficients for G_u - u
            #find poly roots with numpy 
            #poly_roots = fast_polynomial_roots(my_pgf_coef)
            poly_roots = np.roots(np.flip(my_pgf_coef))
            
            filtered_roots = _filter_roots(poly_roots, conditions)  # ensure roots are real and between 0 and 1
            if solve_root:
                return np.min(filtered_roots)
            else:
                return 1 - np.min(filtered_roots)


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
    if perturbation_type == "additive":
        perturbed_coefs = poly_coefs * (1 + delta * alpha_i)
    elif perturbation_type == "multiplicative": 
        perturbed_coefs = poly_coefs * (1 * delta * alpha_i)
    else:
        raise ValueError("Perturbation type must be either additive or multiplicative")
    
    return perturbed_coefs



def l_x_metric(og_roots, perturbed_roots, delta, K, N):
    SCE_list = []
    SCE_list.append(np.abs(og_roots - perturbed_roots) / delta * np.abs(og_roots))
    normed_sce = np.linalg.norm(SCE_list, axis=0)

    return omega(K) / omega(N) * np.mean(normed_sce)


def l_x_algo(
    my_poly_coef,
    is_pgf=True,
    K=1000,
    conditions=None,
    delta=0.001,
    perturbation_type="additive",
    bifurcation=False,
    derivative_test = True
):
    """
    Calculate the stability measure using the Laub-Xia algorithm as outlined in DOI: 10.1137/070702242

    Args:
        my_poly_coef (array-like): Coefficients of the polynomial.
        is_pgf (bool, optional): Whether the polynomial is a probability generating function. Defaults to True.
        K (int, optional): Number of iterations. Defaults to 10.
        conditions (list, optional): List of conditions. Defaults to None.
        delta (float, optional): Perturbation value. Defaults to 0.001.
        perturbation_type (str, optional): Type of perturbation. Defaults to "additive".
        bifurcation (bool, optional): Whether to return the bifurcation list. Defaults to False.
    Returns:
        float or list: The stability measure or the bifurcation list.

    """
    if conditions is None:
        conditions = []
    all_og_roots_conditions = np.empty(K)
    all_perturbed_roots_conditions = np.empty(K)
    N = len(my_poly_coef)
    vec_list = [generate_sphere_point(N) for _ in range(K)]  # Random error
    Z = np.column_stack(vec_list)

    SCE_list = []
    Diff_list = []

    # Root solving and error
    for i in range(K):
        og_roots = fast_polynomial_roots(my_poly_coef)
        #og_roots = _solve_self_consistent_equation(my_poly_coef, conditions,derivative_test=True) # find the roots of the self consistent equation for the unperturbed degree sequence
        #delta = delta*np.sqrt(norm(og_roots) * np.finfo(float).eps)  # set the delta value for the perturbation, see paper for more details
        #delta = np.sqrt(norm(og_roots) * 2**(-16))
        delta =  2**(-16)
        alpha_i = Z[:, i]  # the random error vector for the ith iteration

        my_perturbed_poly_coefs = _perturb_polynomial(my_poly_coef, delta, alpha_i, perturbation_type)  # perturb the polynomial by the random error vector
        perturbed_roots = _solve_self_consistent_equation(my_perturbed_poly_coefs, conditions, derivative_test = True)  # find the roots of the self consistent equation for the unperturbed degree sequence
        #perturbed_roots = fast_polynomial_roots(my_perturbed_poly_coefs)
        #the corrcet root for the giant component size is the minimum
        # og_roots = np.min(np.real(og_roots))
        # perturbed_roots = np.min(np.real(perturbed_roots))

        SCE_list.append(np.abs(perturbed_roots - og_roots) / delta * np.abs(og_roots))
        Diff_list.append(perturbed_roots - og_roots)


        all_perturbed_roots_conditions[i] = perturbed_roots

        all_og_roots_conditions[i] = og_roots

    normed_sce = np.linalg.norm(SCE_list, axis=0)  # provides the total displacement of all differences.

    if bifurcation:
        return Diff_list
    else:
        return normed_sce
    # return omega(K) / omega(N) * normed_sce

# print('Stop')
# x = fast_polynomial_roots([0.2,0.3,0.5], solve_root= True)
# print(x)