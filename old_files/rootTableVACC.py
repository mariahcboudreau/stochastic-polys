#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:30:33 2023

@author: mariahboudreau


Root table to figure file FOR THE VACC
"""

###########
##
## BASE CODE TO RUN THE INFORMATION FOR THE FIGURES
##
###########

import numpy as np
import pandas as pd
import scipy
import math
import joypy
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as cm
import networkx as nx
from collections import Counter


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




#Actual code to run for the VACC


def constructColTable(r0, k, numSD, inc, maxk):
    
    
    U = np.zeros(numSD)
    standDevs = np.zeros(numSD)
    g1 = np.zeros(maxk)
    g1True = np.zeros(maxk)
    
    d = {'SD of Normal': standDevs, 'Root': U}
    df = pd.DataFrame(data=d)
    df['R'] = r0
    df['K'] = k
    
    a = 1/k
    
    # print(k)
    
    for j in range(numSD):
        
         df['SD of Normal'][j] = j*inc 
         for i in range(maxk):
             
             error = np.random.normal(0,j*inc)
             # while error < -1:
             #     error = np.random.normal(0,j*inc)
                 
             g1True[i] = (math.gamma(i+k)/(math.factorial(i)*math.gamma(k)))*((a*r0)/(1+a*r0))**(i) * (1/(1 + a*r0))**(k)
             while error < -g1True[i]:
                 error = np.random.normal(0,j*inc)
                 
             g1[i] = g1True[i]*(1 + error)
             #g1[i] = g1True[i] + error
             
         g1 = g1/np.sum(g1)
         g1True = g1True/np.sum(g1True)
         
        
         
        #Solving for the pgf
         root, outbreak = pgfOutbreak(g1)
         
         df['Root'][j] = root
         
    return df
    

    
###########
##
## SPECIFIC SITUATIONS FROM NOTES TABLE
## 
##    
###########
    



r0_inst = np.array([3.7, 1.9, 3.9, 2.2, 3.9, 2.6, 2.8, 1.5])
k_inst = np.array([0.14, 0.21, 0.13, 0.17, 0.12, 0.14, 0.11, 0.18])


## CONSTANTS
numSD = 11
inc = 0.01
maxk = 20
    

## SITUATION ONE: Root value = ~0.83
##                  Hetero: R0 = 3.7 k = 0.14
##                  Homo: R0 = 1.9 k = 0.21

## SITUATION TWO: Root value = ~0.85
##                  Hetero: R0 = 3.9 k = 0.13
##                  Homo: R0 = 2.2 k = 0.17


## SITUATION THREE: Root value = ~0.8712
##                  Hetero: R0 = 3.9 k = 0.12
##                  Homo: R0 = 2.6 k = 0.14

## SITUATION FOUR: Root value = ~0.915
##                  Hetero: R0 = 2.8 k = 0.11
##                  Homo: R0 = 1.5 k = 0.18


frames = []
for i in range(len(r0_inst)): 
   tempDF = constructColTable(r0_inst[i], k_inst[i], numSD, inc, maxk) 
   frames.append(tempDF)

finalDf = pd.concat(frames)

finalDf.write.csv("root_table_simulation_results_%d.csv" % myjob)




