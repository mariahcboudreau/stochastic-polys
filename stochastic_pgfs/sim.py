import numpy as np
from numpy.linalg import norm, eig
from scipy.linalg import schur
from scipy.special import factorial2
import scipy
import matplotlib.pyplot as plt
import random
import math

from stochastic_pgfs.pgfs import *
from stochastic_pgfs.laub_xia_algo import kappa_SCE, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.stochastic_poly import *

####
#
#   Class defining PGFs, sets up coefficients and determines derivatives
#
####

# class PGF:
#     def __init__(self, coef):
#         self.coef = np.array(coef, dtype=float)
    
#     def __call__(self, x):
#         return sum(self.coef[i] * x**i for i in range(len(self.coef)))
    
#     def derivative(self):
#         deriv_coefs = self.coef[1:] * np.arange(1, len(self.coef))
#         return PGF(deriv_coefs)
    

# def variance_polys(my_poly_coef, num_sims, K=10, conditions=None, delta=0.001):
#     if conditions is None:
#         conditions = []
#     SCE_list = []
#     N = len(my_poly_coef)
#     vec_list = [generate_sphere_point(N) for _ in range(K)]
#     Z = np.column_stack(vec_list)
#     all_perturbed_roots = np.zeros((num_sims, K))
#     for s in range(num_sims):
#         for i in range(K):
#             og_roots = polynomial_roots(my_poly_coef)
#             all_conditions = np.array([True] * len(og_roots))
#             if conditions:
#                 all_conditions = np.logical_and.reduce([cond(og_roots) for cond in conditions])
#             og_roots = og_roots[all_conditions]
#             delta = np.sqrt(norm(og_roots) * np.finfo(float).eps)
#             perturbed_coefs = my_poly_coef * (1 + delta * Z[:, i])
#             try:
#                 all_perturbed_roots[s,i] = polynomial_roots(perturbed_coefs)[all_conditions]
#             except:
#                 all_perturbed_roots[s,i] = 1
    
#     root_vars = all_perturbed_roots.var(axis = 1)
#     return root_vars.mean()  # Simplified to return the mean as a scalar


######
#Checked on 1/19/22 - works
######
def derivative(p_kCoeff):
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


########
# Do not neet to use this anywhere since we are not truncating anymore
########    
# def adjustedPoly(pgfCoeff, trunc): 
#     preG1 = np.array(pgfCoeff)
    
   
#     #Truncating 
#     c = 0
#     for p in preG1:
#         if p <= trunc:
#             preG1[c] = trunc
#             #print("Trunc occurs at %.4f" %trunc)
#         c += 1    
    
#     # Renormalize after truncation
#     preG1 = preG1/np.sum(preG1)
#     # print("After trunc")
#     # print(preG1)
    
#     if preG1.size > 11:
#         preG1 = preG1[0:11]
#         preG1 = preG1/np.sum(preG1)
#     else:
#         s = preG1.size
#         extra = 11 - s
#         for add in range(extra):
#             np.append(preG1, trunc)
#         preG1 = preG1/np.sum(preG1)    
    
#     # print("After setting to 11")
#     # print(preG1)
    
    
#     return preG1

######
# Checked on 1/20/22 - works
######

def ERCoeff(maxk, prob, n):
    binomCoeff = []
    
    for k in range(maxk):
        binomCoeff.append(scipy.special.binom(n-1, k)*prob**(k)*(1-prob)**(n-1-k))
    
    return binomCoeff

######
# Checked on 1/20/22 - works
######

def ER_FSE_Sim(n, prob):
    G = nx.erdos_renyi_graph(n, prob)
    
    # Calculate N_k frequencies

    degrees = []
    for v in nx.nodes(G):
        degrees.append(nx.degree(G, v))
        
    N_k = []    
    for degVal in range(max(degrees)): #
        N_k.append(degrees.count(degVal))
    
    p_k_sim = np.true_divide(N_k, n)  
    
    #Normalize
    if sum(p_k_sim) == 0:
        return [0,0], G
    else:
        p_k_sim = np.true_divide(p_k_sim, sum(p_k_sim))
        # These probailities are given in the order of ascending degree
    
    
    
    
    return p_k_sim, G


#####
# Checkin on 1/20/22 - works
    
# Used in the pgfOutbreak function
#####

def selfConsistant(g1):
    #give dx
    g1_prime = derivative(g1)
    
    # Make a copy of g1 in the right way
    g1_copy = np.array([x for x in g1])
    
    if g1_copy.size > 1:
        if g1_copy.size > 2: 
            g1_copy[1] = g1_copy[1] - 1
        elif g1_copy.size > 1:
            g1_copy = np.append(g1_copy, -1)
    else:
        g1_copy = np.array([0,-1])
    
    
    g1_copy = np.flip(g1_copy)

    
    if sum(g1_prime) > 1:
        #print("There is a root below 1")
        all_roots = np.roots(g1_copy)
        
        all_roots = all_roots[np.isreal(all_roots)]
        all_roots = all_roots.real
        all_roots = all_roots[all_roots >= 0]
        all_roots = all_roots[all_roots < .99999999999]
        if all_roots.size > 1:
            all_roots = max(all_roots)
        if all_roots.size == 0:
            all_roots = 1
        
    else:
        all_roots = 1
    
    return all_roots
    
    
######
# Checked on 1/20/22 - works
######

def pgfOutbreak(g1Outbreak, infinite):
    
    
    root = selfConsistant(g1Outbreak)
    
    outbreak = 1-root
    
    return root, outbreak

######
# Checked on 1/20/22 - 
# Used in both winkler and winklerTrunc to as the conversion factor
######

def convertFactor(n, pks):
 
    
    if pks.size > 1:
            if pks.size >= 2: 
                pks[1] = pks[1] - 1
               
            else:
                pks = np.append(pks, -1)
                
    else:
        pks = np.array([0,-1])
   
    
    pks = abs(pks)
    
    components = np.empty(len(pks))
    start = 0
    for p in pks:
        if p == 0:
            components[start] = 0
        else:
            components[start] = np.sqrt(np.abs(np.true_divide((1-p), n*p)))
        start += 1
    
    
    
    coeffError = np.sum(components)
    
    
    
    return coeffError


def additiveConversion(pks, sigma):
    inverse = np.true_divide(1,pks)
    value = sigma*inverse*np.sqrt(2/math.pi)
    
    return np.divide(1, np.sum(value))

def multiplicativeConversion(pks, sigma):
    
    term = np.sqrt(2/math.pi)
    
    return np.true_divide(1,len(pks)*sigma*term)
    

#####
# Checked on 1/20/22 - works
# Changing the coefficients to alter them for the log function 
#####    


def prepLogFunction(coeffs):
    small = min(np.abs(coeffs))
    
    coeffs= np.delete(coeffs , np.where(coeffs == small))
    
    coeffs_fix = np.true_divide(small, coeffs)
    
    return coeffs_fix, small


######
# Checked on 1/20/22 - working correctly
######

def winklerTrunc(n, true_root, true_pk_coeff, noisyCoef, sigma):
    # 
    # n = population
    # maxk = max degree
    # true_root = the root from the PGF
    # prime_coeff = coefficients of the derivative function
    # true_pk_coeff = the coefficients of the polynomial G1
    
    a_i = np.array(true_pk_coeff)
    a_plus_error = np.array(noisyCoef)
   
    
    # making phi a vector the same size as the coefficients
    phi = np.zeros(a_i.size)
    for k in range(a_i.size):
        phi[k] = true_root**k

    
    
    conversion = 0
    

    if a_i.size == 0:
        winklerMeasure = 0
    elif a_i.size == 1 and a_i[0] == 0:
        winklerMeasure = 0
    else:
            
         if a_i.size > 1:
            if a_i.size >= 2: 
                a_i[1] = a_i[1] - 1
                a_plus_error[1] = a_plus_error[1] - 1
                prime_coeff = derivative(a_i)
            else:
                a_i = np.append(a_i, -1)
                a_plus_error = np.append(a_plus_error, -1)
                prime_coeff = derivative(a_i)
         else:
            a_i = np.array([0,-1])
            prime_coeff = np.array([-1])    
        
         a_i = a_i[a_i != 0] #takes out the zeros for the Winkler computation

         
         
         aInverse, smallCoef = prepLogFunction(a_i)
         
         #aInverse, smallCoef = prepLogFunction(a_plus_error)
        
         prime_coeff = np.flip(prime_coeff)
         prime = np.poly1d(prime_coeff)
        
         
         #conversion = convertFactor(n, a_plus_error) FSE
         
         conversion = multiplicativeConversion(a_i, sigma)
   
         #conversion = additiveConversion(a_i, sigma)
        
        
            # Check to see if this measure if positive or negative
         if prime(true_root) == 0:
             winklerMeasure = 0
         else: 
            # winklerMeasure = np.log((1/np.abs(prime(true_root)))* np.linalg.norm(phi, 2)/(true_root * np.linalg.norm((aInverse - smallCoef), 1))*conversion)
             winklerMeasure = -np.log(np.abs(prime(true_root))) + np.log(np.linalg.norm(phi,2)) - np.log(true_root) - np.log(np.linalg.norm(aInverse,1)) + np.log(smallCoef) + np.log(conversion)
    
    
    return np.exp(winklerMeasure)


#####
# Checked on 1/20/22 - works
# Plot is not used as of now 1/20
#####

def violins(roots):
    
    df_roots = pd.DataFrame(roots)
    
    
    plt.figure(figsize = (7,7))
    plt.title("PGF Sensitivity")
    plt.xlabel("Standard Deviation (times 0.01)")
    plt.ylabel("Root U")
    ax = sns.boxplot(data=df_roots, color = "seagreen") 
    plt.show(ax)
