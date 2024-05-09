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

# for i in range(len(r0_inst)): 
#    roots_inst[i], slopeOfSD[i] , finalSD[i], expectedU[i] = output_all(r0_inst[i], k_inst[i], col, numSD, inc, maxk) 
   
#%% FOR VACC
   
import numpy as np  
import pandas as pd
from glob import glob

numSD = 11
inc = 0.01
maxk = 20



#Reading files 
c_dfs = []
for outputfile in glob("./csvs/coeff_table_simulation_results_*.csv"): # df['r'] = 3 for the first run
    c_df = pd.read_csv(outputfile, index_col = 0)
    c_dfs.append(c_df)
    
c_df = pd.concat(c_dfs)


sit_one_coeffs_hetero = c_df[(c_df['R'] == 3.7) & (c_df['K'] == 0.14)]
sit_one_coeffs_homo = c_df[(c_df['R'] == 1.9) & (c_df['K'] == 0.21)]

sit_two_coeffs_hetero = c_df[(c_df['R'] == 3.9) & (c_df['K'] == 0.13)]
sit_two_coeffs_homo = c_df[(c_df['R'] == 2.2) & (c_df['K'] == 0.17)]

sit_three_coeffs_hetero = c_df[(c_df['R'] == 3.9) & (c_df['K'] == 0.12)]
sit_three_coeffs_homo = c_df[(c_df['R'] == 2.6) & (c_df['K'] == 0.14)]

sit_four_coeffs_hetero = c_df[(c_df['R'] == 2.8) & (c_df['K'] == 0.11)]
sit_four_coeffs_homo = c_df[(c_df['R'] == 1.5) & (c_df['K'] == 0.18)]


dfs = []
for outputfile in glob("./csvs/root_table_simulation_results_*.csv"): # df['r'] = 3 for the first run
    df = pd.read_csv(outputfile, index_col = 0)
    dfs.append(df)
    
df = pd.concat(dfs)

sit_one_df_hetero = df[(df['R'] == 3.7) & (df['K'] == 0.14)]
sit_one_df_homo = df[(df['R'] == 1.9) & (df['K'] == 0.21)]

sit_two_df_hetero = df[(df['R'] == 3.9) & (df['K'] == 0.13)]
sit_two_df_homo = df[(df['R'] == 2.2) & (df['K'] == 0.17)]

sit_three_df_hetero = df[(df['R'] == 3.9) & (df['K'] == 0.12)]
sit_three_df_homo = df[(df['R'] == 2.6) & (df['K'] == 0.14)]

sit_four_df_hetero = df[(df['R'] == 2.8) & (df['K'] == 0.11)]
sit_four_df_homo = df[(df['R'] == 1.5) & (df['K'] == 0.18)]

sit_one_df_hetero = sit_one_df_hetero.reset_index()
sit_one_df_hetero = sit_one_df_hetero.drop('index', axis=1)
sit_one_df_homo = sit_one_df_homo.reset_index()
sit_one_df_homo = sit_one_df_homo.drop('index', axis=1)

sit_two_df_hetero = sit_two_df_hetero.reset_index()
sit_two_df_hetero = sit_two_df_hetero.drop('index', axis=1)
sit_two_df_homo = sit_two_df_homo.reset_index()
sit_two_df_homo= sit_two_df_homo.drop('index', axis=1)

sit_three_df_hetero = sit_three_df_hetero.reset_index()
sit_three_df_hetero = sit_three_df_hetero.drop('index', axis=1)
sit_three_df_homo = sit_three_df_homo.reset_index()
sit_three_df_homo = sit_three_df_homo.drop('index', axis=1)

sit_four_df_hetero = sit_four_df_hetero.reset_index()
sit_four_df_hetero = sit_four_df_hetero.drop('index', axis=1)
sit_four_df_homo = sit_four_df_homo.reset_index()
sit_four_df_homo = sit_four_df_homo.drop('index', axis=1)

## STANDARD DEVIATIONS for all situations

sit_one_hetero_standDevs = np.zeros((numSD))
sit_one_homo_standDevs = np.zeros((numSD))

sit_two_hetero_standDevs = np.zeros((numSD))
sit_two_homo_standDevs = np.zeros((numSD))

sit_three_hetero_standDevs = np.zeros((numSD))
sit_three_homo_standDevs = np.zeros((numSD))

sit_four_hetero_standDevs = np.zeros((numSD))
sit_four_homo_standDevs = np.zeros((numSD))

frames = [sit_one_df_hetero, sit_one_df_homo, sit_two_df_hetero, sit_two_df_homo, sit_three_df_hetero, sit_three_df_homo, sit_four_df_hetero, sit_four_df_homo]

for i, sd in zip(range(numSD), np.arange(0,0.11,0.01)):
    
    sit_one_hetero_standDevs[i] = np.std(sit_one_df_hetero.loc[sit_one_df_hetero['SD of Normal'] == sd]['Root'])
    sit_one_homo_standDevs[i] = np.std(sit_one_df_homo.loc[sit_one_df_homo['SD of Normal'] == sd]['Root'])
    sit_two_hetero_standDevs[i] = np.std(sit_two_df_hetero.loc[sit_two_df_hetero['SD of Normal'] == sd]['Root'])
    sit_two_homo_standDevs[i] = np.std(sit_two_df_homo.loc[sit_two_df_homo['SD of Normal'] == sd]['Root'])
    sit_three_hetero_standDevs[i] = np.std(sit_three_df_hetero.loc[sit_three_df_hetero['SD of Normal'] == sd]['Root'])
    sit_three_homo_standDevs[i] = np.std(sit_three_df_homo.loc[sit_three_df_homo['SD of Normal'] == sd]['Root'])
    sit_four_hetero_standDevs[i] = np.std(sit_four_df_hetero.loc[sit_four_df_hetero['SD of Normal'] == sd]['Root'])
    sit_four_homo_standDevs[i] = np.std(sit_four_df_homo.loc[sit_four_df_homo['SD of Normal'] == sd]['Root'])
    
finalSD = np.zeros(8)

finalSD[0] = sit_one_hetero_standDevs[10]
finalSD[1] = sit_one_homo_standDevs[10]
finalSD[2] = sit_two_hetero_standDevs[10]
finalSD[3] = sit_two_homo_standDevs[10]
finalSD[4] = sit_three_hetero_standDevs[10]
finalSD[5] = sit_three_homo_standDevs[10]
finalSD[6] = sit_four_hetero_standDevs[10]
finalSD[7] = sit_four_homo_standDevs[10]



sdValues = np.arange(0, 0.11, 0.01)

    
## SLOPES for all situations

sit_one_hetero_slopes = slopes(sit_one_hetero_standDevs, sdValues, int(sit_one_df_hetero['R'][0]), int(sit_one_df_hetero['K'][0]))

sit_one_homo_slopes = slopes(sit_one_homo_standDevs, sdValues, int(sit_one_df_homo['R'][0]), int(sit_one_df_homo['K'][0]))

sit_two_hetero_slopes = slopes(sit_two_hetero_standDevs, sdValues, int(sit_two_df_hetero['R'][0]), int(sit_two_df_hetero['K'][0]))

sit_two_homo_slopes = slopes(sit_two_homo_standDevs, sdValues, int(sit_two_df_homo['R'][0]), int(sit_two_df_homo['K'][0]))

sit_three_hetero_slopes = slopes(sit_three_hetero_standDevs, sdValues, int(sit_three_df_hetero['R'][0]), int(sit_three_df_hetero['K'][0]))

sit_three_homo_slopes = slopes(sit_three_homo_standDevs, sdValues, int(sit_three_df_homo['R'][0]), int(sit_three_df_homo['K'][0]))

sit_four_hetero_slopes = slopes(sit_four_hetero_standDevs, sdValues, int(sit_four_df_hetero['R'][0]), int(sit_four_df_hetero['K'][0]))

sit_four_homo_slopes = slopes(sit_four_homo_standDevs, sdValues, int(sit_four_df_homo['R'][0]), int(sit_four_df_homo['K'][0]))


slopes_vacc = np.zeros(8)

slopes_vacc[0] = sit_one_hetero_slopes
slopes_vacc[1] = sit_one_homo_slopes
slopes_vacc[2] = sit_two_hetero_slopes
slopes_vacc[3] = sit_two_homo_slopes
slopes_vacc[4] = sit_three_hetero_slopes
slopes_vacc[5] = sit_three_homo_slopes
slopes_vacc[6] = sit_four_hetero_slopes
slopes_vacc[7] = sit_four_homo_slopes

## EXPECTATION OF U for each situation

sit_one_hetero_expect = expectationOfU(np.array(sit_one_coeffs_hetero['True Coef']), sit_one_df_hetero['Root'][0], 0.1)

sit_one_homo_expect = expectationOfU(np.array(sit_one_coeffs_homo['True Coef']), sit_one_df_homo['Root'][0], 0.1)

sit_two_hetero_expect = expectationOfU(np.array(sit_two_coeffs_hetero['True Coef']), sit_two_df_hetero['Root'][0], 0.1)

sit_two_homo_expect = expectationOfU(np.array(sit_two_coeffs_homo['True Coef']), sit_two_df_homo['Root'][0], 0.1)

sit_three_hetero_expect = expectationOfU(np.array(sit_three_coeffs_hetero['True Coef']), sit_three_df_hetero['Root'][0], 0.1)

sit_three_homo_expect = expectationOfU(np.array(sit_three_coeffs_homo['True Coef']), sit_three_df_homo['Root'][0], 0.1)

sit_four_hetero_expect = expectationOfU(np.array(sit_four_coeffs_hetero['True Coef']), sit_four_df_hetero['Root'][0], 0.1)

sit_four_homo_expect = expectationOfU(np.array(sit_four_coeffs_homo['True Coef']), sit_four_df_homo['Root'][0], 0.1)

expectU = np.zeros(8)

expectU[0] = sit_one_hetero_expect
expectU[1] = sit_one_homo_expect
expectU[2] = sit_two_hetero_expect
expectU[3] = sit_two_homo_expect
expectU[4] = sit_three_hetero_expect
expectU[5] = sit_three_homo_expect
expectU[6] = sit_four_hetero_expect
expectU[7] = sit_four_homo_expect

roots_inst = np_zeros(8)

roots_inst[0] = sit_one_df_hetero['Root'][0]
roots_inst[1] = sit_one_df_homo['Root'][0]
roots_inst[2] = sit_two_df_hetero['Root'][0]
roots_inst[3] = sit_two_df_homo['Root'][0]
roots_inst[4] = sit_three_df_hetero['Root'][0]
roots_inst[5] = sit_three_df_homo['Root'][0]
roots_inst[6] = sit_four_df_hetero['Root'][0]
roots_inst[7] = sit_four_def_homo['Root'][0]



    
 #%% Graphing from VACC jobs

fig, axs = plt.subplots(3)
fig.set_size_inches(6,10)
fig.suptitle("VACC Runs")

# Markers o for hetero and * for homo





for i, c in zip(range(0,8,2), ['b', 'g', 'r', 'c']):
    axs[0].scatter(slopes_vacc[i], finalSD[i], color = c, marker = 'o', label = "Hetero Root %0.3f" % roots_inst[i])
    axs[0].scatter(slope_vacc[i+1], finalSD[i+1], color = c, marker = '*', label = "Homo Root %0.3f" % roots_inst[i + 1])
    axs[1].scatter(slope_vacc[i], expectU[i], color = c, marker = 'o', label = "Hetero Root %0.3f" % roots_inst[i])
    axs[1].scatter(slope_vacc[i+1], expectU[i+1], color = c, marker = '*', label = "Homo Root %0.3f" % roots_inst[i + 1])
    axs[2].scatter(finalSD[i], expectU[i], color = c, marker = 'o', label = "Hetero Root %0.3f" % roots_inst[i])
    axs[2].scatter(finalSD[i+1], expectU[i+1], color = c, marker = '*', label = "Homo Root %0.3f" % roots_inst[i + 1])

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
