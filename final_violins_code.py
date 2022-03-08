#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:50:45 2022

@author: mariahboudreau


OFFICIAL VIOLIN CODE
"""


import numpy as np
import pandas as pd
import scipy
import math
import joypy
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
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


def expectationOfCoeff(a_i_coeff, sigma):
    
    a_i, prime_coeff = prepSelfconsistent(a_i_coeff)
    
    aInverse, smallCoef = prepLogFunction(a_i)
    
    bottomValue = np.log(sigma) + np.log(np.sqrt(2)) - np.log(np.sqrt(math.pi)) + np.log(np.linalg.norm(aInverse,1))
    
    return np.exp(bottomValue)




def slopes(rise, runValue, r0, k):
# plt.figure(figsize=(7,4))

    m, b = np.polyfit(runValue, rise, 1)

# slopelabel = "Slope = %.4f"%m

# plt.scatter(run, rise)
# plt.plot(run, m*run + b, label = slopelabel)
# plt.title("Change in the SD of roots versus change in SD of error (R0 = %.1f, k = %.2f)"%(r0, k))   
# plt.ylabel("SD of Roots")
# plt.xlabel("SD of Error")
# plt.legend()
    return m


def violins(roots):
    
    df_roots = pd.DataFrame(roots)
    
    
    plt.figure(figsize = (7,7))
    plt.title("PGF Sensitivity")
    plt.xlabel("Standard Deviation (times 0.01)")
    plt.ylabel("Root U")
    ax = sns.boxplot(data=df_roots, color = "seagreen") 
    plt.show(ax)


def main(r0, k, col, numSD, inc, maxk):
    
    U = np.zeros((col, numSD))
    S = np.zeros((col, numSD))
    winklerMeasures = np.zeros((col, numSD))
    standDevs = np.zeros(numSD)
    g1 = np.zeros(maxk)
    g1True = np.zeros(maxk)
    
    trial = np.zeros(maxk)
    
    a = 1/k
    
    # print(k)
    
    for j in range(numSD):
        
        
        
        for n in range(col):
            
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
            
            #trial = np.divide((g1True - g1), g1True)
            
            
            # if k < 0.12:
            #     print(np.std(trial), j*inc, k)
            
            
           #Solving for the pgf
            root, outbreak = pgfOutbreak(g1)
            
            U[n,j] = root
            S[n,j] = outbreak
            
            
            
           # winklerMeasures[n,j] = winklerTrunc(10000, U[0,0], g1True, g1, j*inc)
            
            #winklerMeasures[n,j] = originalWinklerValue(U[0,0], g1True, g1)
            
     
    for b in range(numSD):    
        standDevs[b] = np.std(U[:,b])
        

      
        
    sdValues = np.arange(numSD)
    sdValues = sdValues*inc
    
    #print(slopes(standDevs, sdValues, r0, k))
    #print(standDevs)
    #violins(U)
    #ridgePlots(U)    
    return U[0,0]


def solvingForExpectation(r0, k, sigma):
    maxk = 20
    
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
    
    expectU = expectationOfU(g1True, true_root, sigma)
    
    return expectU

        
    
#%%% Constructing values for contour plot    
    
col = 100
numSD = 10
inc = 0.01
maxk = 20

r0_vec = np.arange(1, 5, 0.1)
k_vec = np.arange(0.01, 1, 0.01)

r0_vals = np.empty((len(r0_vec)*len(k_vec)))
k_vals = np.empty((len(r0_vec)*len(k_vec)))
root_vals = np.empty((len(r0_vec)*len(k_vec)))

count = 0
for r0 in r0_vec:
    for k in k_vec:
            
        r0_vals[count] = r0
        k_vals[count] = k
        root_vals[count] = main(r0, k, col, numSD, inc, maxk)
        count += 1

df_roots = pd.DataFrame({'R0': r0_vals, 'k':k_vals, 'Roots':root_vals}) 


#%%%% Constructing the values for the heatmap and the heat map


    
sigma = 0.1

r0_vec = np.arange(1, 5, 0.1)
k_vec = np.arange(0.01, 1, 0.01)

r0_vals = np.empty((len(r0_vec)*len(k_vec)))
k_vals = np.empty((len(r0_vec)*len(k_vec)))
expect_root_vals= np.empty((len(r0_vec)*len(k_vec)))

count = 0
for r0 in r0_vec:
    for k in k_vec:
            
        r0_vals[count] = r0
        k_vals[count] = k
        expect_root_vals[count] = solvingForExpectation(r0, k, sigma)
        count += 1

df_delta_roots = pd.DataFrame({'R0': r0_vals, 'k':k_vals, 'Expectation':expect_root_vals}) 

heatmap_plot = df_delta_roots.pivot(index = 'k', columns = 'R0', values = 'Expectation' )
sns.set(rc={'text.usetex': True})

sns.heatmap(heatmap_plot, cmap = "Greens")

#%%%% Contour plot


pivot = df_roots.pivot('k', 'R0', 'Roots')

X = pivot.columns.values
Y = pivot.index.values
Z = pivot.values

x,y = np.meshgrid(X, Y)

levels = np.arange(0, 1, 0.025)

fig, ax = plt.subplots()
cs = ax.contour(x, y, Z, levels=levels)



plt.colorbar(cs)


plt.clabel(cs, inline=1, fontsize=8)

plt.xlabel("$R_{0}$")
plt.ylabel("k, dispersion parameter")

plt.imshow()



#%%%% Retrieving the vertices from the contour plots

vertice_sets = cs.collections[0].get_paths()

plt.bar(vertices_sets, expectations)

