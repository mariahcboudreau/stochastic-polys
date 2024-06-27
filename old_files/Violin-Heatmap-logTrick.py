#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Violin Plots and Heatamap code with Log trick

@author: mariahboudreau
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

#####
# Checked on 1/20/22 - works
# Used for accenting the boxplots and used as of now 1/20
#####

def ridgePlots(roots):
    df_roots = pd.DataFrame(roots)
    
    joy = joypy.joyplot(df_roots, grid="y", linewidth=1, legend=False, figsize=(7,4), colormap=cm.Greens)
    
    plt.title("PGF Sensitivity")
    plt.ylabel("Standard Deviation (times 0.01)")
    plt.xlabel("Root U")
    plt.show(joy)

#####
# Checked on 1/20/22 - works
# Used for Change in the SD of roots versus change in SD of error plots
#####

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
            root, outbreak = pgfOutbreak(g1, True)
            
            U[n,j] = root
            S[n,j] = outbreak
            
            
            
            winklerMeasures[n,j] = winklerTrunc(10000, U[0,0], g1True, g1, j*inc)
            
            #winklerMeasures[n,j] = originalWinklerValue(U[0,0], g1True, g1)
            
     
    for b in range(numSD):    
        standDevs[b] = np.std(U[:,b])
        

      
    standDevs_length = len(standDevs)
    last_element = standDevs[standDevs_length - 1]
    
    sdValues = np.arange(numSD)
    sdValues = sdValues*inc
    print("Last SD value: %.5f" %last_element)
    print("Slope: %.5f"% (slopes(standDevs, sdValues, r0, k)))
    #print(standDevs)
    violins(U)
    #ridgePlots(U)    
    return U   
    

# def originalWinklerValue(true_root, true_pk_coeff, noisyCoef):
    
#     a_i = np.array(true_pk_coeff)
#     a_plus_error = np.array(noisyCoef)
   
    
#     # making phi a vector the same size as the coefficients
#     phi = np.zeros(a_i.size)
#     for k in range(a_i.size):
#         phi[k] = true_root**k


    

#     if a_i.size == 0:
#         winklerMeasure = 0
#     elif a_i.size == 1 and a_i[0] == 0:
#         winklerMeasure = 0
#     else:
            
#          if a_i.size > 1:
#             if a_i.size >= 2: 
#                 a_i[1] = a_i[1] - 1
#                 a_plus_error[1] = a_plus_error[1] - 1
#                 prime_coeff = derivative(a_i)
#             else:
#                 a_i = np.append(a_i, -1)
#                 a_plus_error = np.append(a_plus_error, -1)
#                 prime_coeff = derivative(a_i)
#          else:
#             a_i = np.array([0,-1])
#             prime_coeff = np.array([-1])    
        
#          a_i = a_i[a_i != 0] #takes out the zeros for the Winkler computation

#          prime_coeff = np.flip(prime_coeff)
#          prime = np.poly1d(prime_coeff)
         
         
#          aInverse, smallCoef = prepLogFunction(a_i)
    
#          # Check if this is negative  
#          measure = -np.log(np.abs(prime(true_root))) + np.log(np.linalg.norm(phi,2)) - np.log(true_root) - np.log(np.linalg.norm(aInverse,1)) + np.log(smallCoef)
    
#     return np.exp(measure)

            
#%% Running violins
        
r0 = 1.9
k = 0.21
col = 100
numSD = 10
inc = 0.01
maxk = 20
# need to incorporate the noisy values into the measure
UValues = main(r0, k, col, numSD, inc, maxk)





#%% Running multiple k values for a heat map for simulation reasons

r0 = 1.1
k = np.linspace(0.01,.1, num = 50)
col = 100
numSD = 20
inc = 0.01
maxk = 20

varyingK = np.empty((50))
index = 0
for value in k:
    temp = main(r0, value, col, numSD, inc, maxk)
    varyingK[index] = temp
    index += 1

plt.title("Change in Standard Deviation of the Roots over Standard Deviation of Error varying over average degree")
plt.xlabel("k values")
plt.ylabel("Slopes")
plt.scatter(k, varyingK)

#%% Heat map



"""
Heat maps

"""



def mainWithTrueCoef(r0, k, col, numSD, inc, maxk):
    
    U = np.zeros((col, numSD))
    S = np.zeros((col, numSD))
    compSens = np.zeros((col, numSD))
    avgComp = np.zeros(col)
    g1True = np.zeros(maxk)
    g1= np.zeros(maxk)
    
    a = 1/k
    
    for j in range(numSD):
        
        
        for n in range(col):
            
            for i in range(maxk):
                
                error = np.random.normal(0,j*inc)
                while error < -1:
                    error = np.random.normal(0,j*inc)
                    
                g1True[i] = (math.gamma(i+k)/(math.factorial(i)*math.gamma(k)))*((a*r0)/(1+a*r0))**(i) * (1/(1 + a*r0))**(k)
                g1[i] = g1True[i]*(1+error)
                
            g1 = g1/np.sum(g1)
            g1True = g1True/np.sum(g1True)
            
            
            
           #Solving for the pgf
            root, outbreak = pgfOutbreak(g1, True)
            
            U[n,j] = root
            S[n,j] = outbreak
            
            
            #compSens[n,j] = originalWinklerValue(U[0,0], g1True, g1)
            compSens[n,j] = winklerTrunc(10000, U[0,0], g1True, g1, j*inc)
            
            
    
    for b in range(col):
        avgComp[b] = np.sum(compSens[b,:])/numSD  
        
    return avgComp

def heatmap(dim, col, numSD, inc, maxk):
    
    rh = 2/dim
    kh = .15/dim
    
    M = (dim-2)**2
    
    r0_values = np.zeros(M)
    k_values = np.zeros(M)
    compWinkler = np.zeros(M)
    
    counter = 0
    
    for b in range(1,dim-1):
        
        r = 2+rh*b # Value for R0 for each iteration
        
        for c in range(1,dim-1):
            
            kLoop = kh*c # Value for k for each iteration
            
            
            
            avgCompSens = mainWithTrueCoef(r, kLoop, col, numSD, inc, maxk)
            
            r0_values[counter] = r
            k_values[counter] = kLoop
            compWinkler[counter] = np.sum(avgCompSens)/len(avgCompSens)
            counter +=1
            
    df_winkler = pd.DataFrame({'R0': r0_values, 'k':k_values, 'Winkler':np.exp(compWinkler)}) 
    return df_winkler        
            
          
#%%%    Running heatmap     
        
            
dim = 15
col = 10
numSD = 10
inc = 0.01
maxk = 20

df_Heatmap = heatmap(dim, col, numSD, inc, maxk)


data_2_plot = df_Heatmap.pivot(index = 'k', columns = 'R0', values = 'Winkler' )
sns.heatmap(data_2_plot, cmap = "Greens")

    

#%%% Change the number of SD, keep the error constant



def mainWithTrueCoefNonSD(r0, k, col, sigma, maxk):
    
    U = np.zeros((col))
    S = np.zeros((col))
    compSens = np.zeros(col)
    avgComp = np.zeros(col)
    g1True = np.zeros(maxk)
    g1= np.zeros(maxk)
    
    a = 1/k
    
    
   
        
    for n in range(col):
        
        for i in range(maxk):
            
            error = np.random.normal(0,sigma)
            
                
            g1True[i] = (math.gamma(i+k)/(math.factorial(i)*math.gamma(k)))*((a*r0)/(1+a*r0))**(i) * (1/(1 + a*r0))**(k)
            
            while error < -g1True[i]:
                error = np.random.normal(0,sigma)
                
            g1[i] = g1True[i]*(1+error)
            
        g1 = g1/np.sum(g1)
        g1True = g1True/np.sum(g1True)
        
        
        
        #Solving for the pgf
        root, outbreak = pgfOutbreak(g1, True)
        
        U[n] = root
        S[n] = outbreak
        
        
        
        #compSens[n,j] = originalWinklerValue(U[0,0], g1True, g1)
        compSens[n] = winklerTrunc(10000, U[0], g1True, g1, sigma)
        
   
    
    # for b in range(col):
    #     avgComp[b] = np.sum(compSens[b,:])/numSD  
        
    return compSens

def heatmapNonSD(dim, col, sigma, maxk):
    
    rh = 2/dim
    kh = .15/dim
    
    M = (dim-2)**2
    
    r0_values = np.zeros(M)
    k_values = np.zeros(M)
    compWinkler = np.zeros(M)
    
    counter = 0
    
    for b in range(1,dim-1):
        
        r = 2+rh*b # Value for R0 for each iteration
        
        for c in range(1,dim-1):
            
            kLoop = 0.1 + kh*c # Value for k for each iteration
            
            
            
            CompSens = mainWithTrueCoefNonSD(r, kLoop, col, sigma, maxk)
            
            r0_values[counter] = r
            k_values[counter] = kLoop
            compWinkler[counter] = np.sum(CompSens)/len(CompSens)
            counter +=1
            
    df_winkler = pd.DataFrame({'R0': r0_values, 'k':k_values, 'Winkler':compWinkler}) 
    return df_winkler 


#%%%    Running heatmap with constant error   
        
            
dim = 25
col = 10
sigma = 0.2
maxk = 100

df_Heatmap = heatmapNonSD(dim, col, sigma, maxk)


data_2_plot = df_Heatmap.pivot(index = 'k', columns = 'R0', values = 'Winkler' )
sns.heatmap(data_2_plot, cmap = "Greens")

    
