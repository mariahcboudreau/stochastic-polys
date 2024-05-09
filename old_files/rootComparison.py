#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:30:33 2023

@author: mariahboudreau


Root table to figure file
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




def slopes(rise, runValue, r0, k): #rise is for the sd of the violins and then run is the increments of changing sigma
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
    plt.ylim(0.55, 0.95)
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
            
            #print(np.sd(g1))
            # Make sure that the standard deviation is what we are saying it is
            
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
    
    print(sdValues[10])
    print(standDevs[10])
    
    print(slopes(standDevs[:10], sdValues[:10], r0, k))
    #print(standDevs)
    #violins(U)
    #ridgePlots(U)    
    return U[0,0]


def solvingForExpectation(params, sigma):
    maxk = 20
    
    r0 = params[0]
    k = params[1]
    
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
    
    return expectU #same thing as delta root


def output_all(r0, k, col, numSD, inc, maxk):
    
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
            
            #print(np.sd(g1))
            # Make sure that the standard deviation is what we are saying it is
            
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
    
    # print(sdValues[10])
    # print(standDevs[10])
    
    #print(slopes(standDevs[:10], sdValues[:10], r0, k))
    #print(standDevs)
    #violins(U)
    #ridgePlots(U)    
    return U[0,0], slopes(standDevs[:numSD-1], sdValues[:numSD-1], r0, k), standDevs[numSD-1], expectationOfU(g1True,U[0,0], standDevs[numSD-1])


#%% 
    
###########
##
## SPECIFIC SITUATIONS FROM NOTES TABLE
## 
##    
###########
    
## STORE VALUES

r0_inst = np.array([3.7, 1.9, 3.9, 2.2, 3.9, 2.6, 2.8, 1.5])
k_inst = np.array([0.14, 0.21, 0.13, 0.17, 0.12, 0.14, 0.11, 0.18])

roots_inst = np.zeros(len(r0_inst))
slopeOfSD = np.zeros((len(r0_inst)))
finalSD = np.zeros((len(r0_inst)))
expectedU = np.zeros((len(r0_inst)))

## CONSTANTS

col = 100
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

# for loop hitting all the values 

for i in range(len(r0_inst)): 
   roots_inst[i], slopeOfSD[i] , finalSD[i], expectedU[i] = output_all(r0_inst[i], k_inst[i], col, numSD, inc, maxk) 
   
#%% FOR VACC
   
import pandas as pd
from glob import glob

#Reading files 
dfs = []
for outputfile in glob("./csvs/root_table_simulation_results_*.csv"): # df['r'] = 3 for the first run
    df = pd.read_csv(outputfile)
    dfs.append(df)
    
df = pd.concat(dfs)


    
#%% 
   
###########
##
## GRAPHING FOR COMPARISON (ALL)
##
###########
   
   
fig, axs = plt.subplots(3)
fig.set_size_inches(6,10)

# Markers o for hetero and * for homo

for i, c in zip(range(0,8,2), ['b', 'g', 'r', 'c']):
    axs[0].scatter(slopeOfSD[i], finalSD[i], color = c, marker = 'o', label = "Hetero Root %0.3f" % roots_inst[i])
    axs[0].scatter(slopeOfSD[i+1], finalSD[i+1], color = c, marker = '*', label = "Homo Root %0.3f" % roots_inst[i + 1])
    axs[1].scatter(slopeOfSD[i], expectedU[i], color = c, marker = 'o', label = "Hetero Root %0.3f" % roots_inst[i])
    axs[1].scatter(slopeOfSD[i+1], expectedU[i+1], color = c, marker = '*', label = "Homo Root %0.3f" % roots_inst[i + 1])
    axs[2].scatter(finalSD[i], expectedU[i], color = c, marker = 'o', label = "Hetero Root %0.3f" % roots_inst[i])
    axs[2].scatter(finalSD[i+1], expectedU[i+1], color = c, marker = '*', label = "Homo Root %0.3f" % roots_inst[i + 1])

axs[0].set_xlabel("Slope of SD")
axs[1].set_xlabel("Slope of SD")
axs[2].set_xlabel("Final SD")
axs[0].set_ylabel("Final SD")
axs[1].set_ylabel("Expected Delta U")
axs[2].set_ylabel("Expected Delta U")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


#%% 

###########
##
## GRAPHING FOR COMPARISON (only Final SD and Expected Delta U)
##
###########

fig, ax = plt.subplots()

# Markers o for hetero and * for homo

for i, c in zip(range(0,8,2), ['b', 'g', 'r', 'c']):
    ax.scatter(finalSD[i], expectedU[i], color = c, marker = 'o', label = "Hetero Root %0.3f" % roots_inst[i])
    ax.scatter(finalSD[i+1], expectedU[i+1], color = c, marker = '*', label = "Homo Root %0.3f" % roots_inst[i + 1])
    
ax.axline((0.02,0.02), slope = 1)

plt.xlabel("Final SD")
plt.ylabel("Expected Delta U")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
