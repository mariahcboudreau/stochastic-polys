#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:40:38 2023

@author: mariahboudreau
"""
"""

OFFICIAL VIOLIN CODE FOR VACC
"""


import numpy as np
import pandas as pd
import scipy
import math
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from collections import Counter

import matplotlib as mpl
import os, sys


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


######
# Checked on 1/20/22 - works
######

def ERCoeff(maxk, prob, n):
    binomCoeff = []
    
    for k in range(maxk):
        binomCoeff.append(scipy.special.binom(n-1, k)*prob**(k)*(1-prob)**(n-1-k))
    
    return binomCoeff


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

def pgfOutbreak(g1Outbreak):
    
    
    root = selfConsistant(g1Outbreak)
    
    outbreak = 1-root
    
    return root, outbreak



######
###### WINKLER FUNCTIONS
######

#####
# Checked on 1/20/22 - works
# Changing the coefficients to alter them for the log function 
#####    


def prepLogFunction(coeffs):
    small = min(np.abs(coeffs))
    
    coeffs= np.delete(coeffs , np.where(coeffs == small))
    
    coeffs_fix = np.true_divide(small, coeffs)
    
    return coeffs_fix, small


def prepSelfconsistent(coeffs):
    
    a_array = np.array(coeffs)
    
    if a_array.size > 1:
       if a_array.size >= 2: 
           a_array[1] = a_array[1] - 1
           a_prime_array = derivative(a_array)
       else:
           a_array = np.append(a_array, -1)
           a_prime_array= derivative(a_array)
    else:
       a_array = np.array([0,-1])
       a_prime_array = np.array([-1])    
        
    a_array =  a_array[ a_array != 0] #takes out the zeros for the Winkler computation

    
    return a_array, a_prime_array

def expectationOfU(a_i_coeff, true_root, sigma):
    
    
    a_i, prime_coeff = prepSelfconsistent(a_i_coeff)
    
    
    # making phi a vector the same size as the coefficients
    phi = np.zeros(a_i.size)
    for k in range(a_i.size):
        phi[k] = true_root**k

   
    prime_coeff = np.flip(prime_coeff)
    prime = np.poly1d(prime_coeff)
    
    
    
    value = -np.log(true_root) - np.log(np.abs(prime(true_root))) + np.log(np.sqrt(2)) + np.log(sigma) + np.log(np.linalg.norm(phi,2)) - np.log(np.sqrt(math.pi))
    
    
    return np.exp(value)



def solvingForExpectation(params, sigma):
    maxk = 50
    
    standDevs = np.zeros(1)
    
    
    r0 = params[0]
    k = params[1]
    
    d = {'SD of Normal': standDevs}
    df = pd.DataFrame(data=d)
    df['SD of Normal'] = sigma
    df['R0'] = r0
    df['K'] = k
    
    a = 1/k
    
    g1 = np.zeros(maxk)
    g1True = np.zeros(maxk)
    
    for i in range(maxk):
        error = np.random.normal(0,sigma)
             # while error < -1:
             #     error = np.random.normal(0,j*inc)
                 
        g1True[i] = (math.gamma(i+k)/(math.factorial(i)*math.gamma(k)))*((a*r0)/(1+a*r0))**(i) * (1/(1 + a*r0))**(k)
        while error < -g1True[i]:
             error = np.random.normal(0,sigma)
             
        g1[i] = g1True[i]*(1 + error)
        
         
    g1 = g1/np.sum(g1)
    g1True = g1True/np.sum(g1True)
     
     
    true_root, true_outbreak = pgfOutbreak(g1True)
    #Solving for the pgf
    root, outbreak = pgfOutbreak(g1)
    df['True Root'] = true_root
    df['True Outbreak'] = true_outbreak
    df['Root'] = root
    df['Outbreak'] = outbreak
    
    expectU = expectationOfU(g1True, true_root, sigma)
    
    df['Expection Delta U'] = expectU
    
    coeff = {'True Coef': g1True, 'R': r0, 'K': k}
    coeff = pd.DataFrame(data=coeff)
    
    return df, coeff




# Given a list we want to divide up across (for example) 100 workers,
# which piece of the list is my responsibility?
def getMyChunk(array, rank, group_size):
	my_chunk = []
	for i,row in enumerate(array):
		if( i % group_size == rank ):
			my_chunk.append(row)
	return my_chunk

# Parses CSV file, returns the portion assigned to this worker
def readCSV(filename, rank, group_size):
	with open(filename, "r") as csv:
		lines = csv.read().split("\n")
		tasks = list(map(lambda l: l.split(","), lines))
		return getMyChunk(tasks, rank, group_size)


frames = []
framesCoeff = [] 

rank = int(sys.argv[1])
total_workers = int(sys.argv[2])
myjobs = readCSV("violin_code_vacc_parameters.csv", rank, total_workers)
for (_,r0,k) in myjobs:
    		# Since r0, k came from a csv file, they're all strings
    		# so convert to the correct types here
    
    r0 = float(r0)
    k = float(k)
    sigma = 0.1
    
    tempDF, coeffDF = solvingForExpectation([r0, k], sigma)
    frames.append(tempDF)
    framesCoeff.append(coeffDF)
    
    
    
finalDf = pd.concat(frames, ignore_index = True)
finalDfCoeff = pd.concat(framesCoeff, ignore_index = False)


finalDf.to_csv("heatmap_results_3-13-23_%d.csv" % int(rank))    


