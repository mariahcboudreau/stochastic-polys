#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PGF Sensitivty Violin plots

Mariah Boudreau
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

def adjustedPoly(pgfCoeff, trunc): 
    preG1 = np.array(pgfCoeff)
    
   
    #Truncating 
    c = 0
    for p in preG1:
        if p <= trunc:
            preG1[c] = trunc
            #print("Trunc occurs at %.4f" %trunc)
        c += 1    
    
    # Renormalize after truncation
    preG1 = preG1/np.sum(preG1)
    # print("After trunc")
    # print(preG1)
    
    if preG1.size > 11:
        preG1 = preG1[0:11]
        preG1 = preG1/np.sum(preG1)
    else:
        s = preG1.size
        extra = 11 - s
        for add in range(extra):
            np.append(preG1, trunc)
        preG1 = preG1/np.sum(preG1)    
    
    # print("After setting to 11")
    # print(preG1)
    
    
    return preG1

def ERCoeff(maxk, prob, n):
    binomCoeff = []
    
    for k in range(maxk):
        binomCoeff.append(scipy.special.binom(n-1, k)*prob**(k)*(1-prob)**(n-1-k))
    
    return binomCoeff


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
    
    
    
    
    return p_k_sim, G


def selfConsistant(g1, trunc, infinite):
    
    if infinite == False:
        g1 = adjustedPoly(g1, trunc)
    
    #give dx
    g1_prime = derivative(g1)
    
    
    if g1.size > 1:
        if g1.size > 2: 
            g1[1] = g1[1] - 1
        elif g1.size > 1:
            g1 = np.append(g1, -1)
    else:
        g1 = np.array([0,-1])
    
    
    g1 = np.flip(g1)

    
    if sum(g1_prime) > 1:
        #print("There is a root below 1")
        all_roots = np.roots(g1)
        
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
    

def pgfOutbreak(g1Outbreak, trunc, infinite):
    
    
    root = selfConsistant(g1Outbreak, trunc, infinite)
    
    outbreak = 1-root
    
    return root, outbreak


def convertFactor(n, pks):
    
    # if pks.size == 0:
    #     print("EMPTY")
    # elif pks.size >=1:   
    #     if pks.size == 2:
    #         if pks[1] < 0:
    #             print("NEGATIVE")
    
    
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

def winklerTrunc(n, true_root, true_pk_coeff, noisyCoef, trunc):
    # n = population
    # maxk = max degree
    # true_root = the root from the PGF
    # prime_coeff = coefficients of the derivative function
    # true_pk_coeff = the coefficients of the polynomial G1
    
   
   
    a_i = adjustedPoly(true_pk_coeff, trunc)
    a_plus_error = adjustedPoly(noisyCoef, trunc)
    
    # making phi a vector 
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

         aInverse = np.true_divide(1,a_i) # This is correct for the l1norm
        
         prime_coeff = np.flip(prime_coeff)
         prime = np.poly1d(prime_coeff)
        
         conversion = convertFactor(n, a_plus_error)
   
        
        
        
         if prime(true_root) == 0:
             winklerMeasure = 0
         else: 
             winklerMeasure = (1/np.abs(prime(true_root))) * np.linalg.norm(phi, 2)/(true_root * np.linalg.norm(aInverse, 1))*conversion
    
    #equation 16
    #winklerMeasure = (sigma*np.sqrt(2)/(np.sqrt(math.pi)*true_root))*(1/np.abs(prime(true_root)))*( np.linalg.norm(phi, 2))
   # winklerMeasure = 0
    return winklerMeasure

def  winkler(n, true_root, true_pk_coeff, noisyCoef):
    # n = population
    # maxk = max degree
    # true_root = the root from the PGF
    # prime_coeff = coefficients of the derivative function
    # true_pk_coeff = the coefficients of the polynomial G1
    
    a_i = adjustedPoly(true_pk_coeff, 0)
    a_plus_error = np.array(noisyCoef)
    
    # making phi a vector 
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

         aInverse = np.true_divide(1,a_i) # This is correct for the l1norm
        
         prime_coeff = np.flip(prime_coeff)
         prime = np.poly1d(prime_coeff)
        
         conversion = convertFactor(n, a_plus_error)
   
        
        
         if prime(true_root) == 0:
             winklerMeasure= 0
         else: 
             winklerMeasure = (1/np.abs(prime(true_root))) * np.linalg.norm(phi, 2)/(true_root * np.linalg.norm(aInverse, 1))*conversion
    
    #equation 16
    #winklerMeasure = (sigma*np.sqrt(2)/(np.sqrt(math.pi)*true_root))*(1/np.abs(prime(true_root)))*( np.linalg.norm(phi, 2))
   # winklerMeasure = 0
    return winklerMeasure

def violins(roots):
    
    df_roots = pd.DataFrame(roots)
    
    
    # play around with the figure 
    plt.figure(figsize = (7,7))
    plt.title("PGF Sensitivity")
    plt.xlabel("Standard Deviation (times 0.01)")
    plt.ylabel("Root U")
    ax = sns.boxplot(data=df_roots, color = "seagreen") 
    plt.show(ax)

def ridgePlots(roots):
    df_roots = pd.DataFrame(roots)
    
    joy = joypy.joyplot(df_roots, grid="y", linewidth=1, legend=False, figsize=(7,4), colormap=cm.Greens)
    
    plt.title("PGF Sensitivity")
    plt.ylabel("Standard Deviation (times 0.01)")
    plt.xlabel("Root U")
    plt.show(joy)
    
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
    
    a = 1/k
    
    for j in range(numSD):
        
        
        for n in range(col):
            
            for i in range(maxk):
                
                error = np.random.normal(0,j*inc)
                while error < -1:
                    error = np.random.normal(0,j*inc)
                    
                g1True[i] = (math.gamma(i+k)/(math.factorial(i)*math.gamma(k)))*((a*r0)/(1+a*r0))**(i) * (1/(1 + a*r0))**(k)
                g1[i] = g1True[i]*(1 + error)
                
            g1 = g1/np.sum(g1)
            g1True = g1True/np.sum(g1True)
            
            thresh = 1/10000
           #Solving for the pgf
            root, outbreak = pgfOutbreak(g1, thresh, True)
            
            U[n,j] = root
            S[n,j] = outbreak
            
            
            
            winklerMeasures[n,j] = winklerTrunc(10000, U[0,0], g1True, g1, thresh)
            #winklerMeasures[n,j] = winkler(10000, U[0,0], g1True, g1)
      
    for b in range(numSD):    
        standDevs[b] = np.std(U[:,b])
      
        
    sdValues = np.arange(numSD)
    sdValues = sdValues*inc
    
    return slopes(standDevs, sdValues, r0, k)
    violins(U)
    #ridgePlots(U)    
    #return winklerMeasures    
    
    
            
#%% Running violins
        
r0 = 3.9
k = 0.09
col = 100
numSD = 20
inc = 0.01
maxk = 11
# need to incorporate the noisy values into the measure
winklerValues = main(r0, k, col, numSD, inc, maxk)



#%% Running multiple k values for a heat map for simulation reasons

r0 = 2.3
k = np.linspace(0.1,.2, num = 50)
col = 100
numSD = 20
inc = 0.01
maxk = 11

varyingK = np.empty((50))
index = 0
for value in k:
    temp = main(r0, value, col, numSD, inc, maxk)
    varyingK[index] = temp
    index += 1

plt.title("Change in Standard Deviation of the Roots over Standard Deviation of Error varying dispersion parameter")
plt.xlabel("Dispersion parameter values")
plt.ylabel("Slopes")
plt.scatter(k, varyingK)

#%% Heat map



"""
Heat maps

"""

import numpy as np
import pandas as pd
import scipy
import math
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

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


def ERCoeff(maxk, prob, n):
    binomCoeff = []
    
    for k in range(maxk):
        binomCoeff.append(scipy.special.binom(n-1, k)*prob**(k)*(1-prob)**(n-1-k))
    
    return binomCoeff

#This function will take in the PGF and adjust it into the polynomial we want to use to solve for the root/use for Winkler measure
def adjustedPoly(pgfCoeff, trunc): 
    preG1 = np.array(pgfCoeff)
    
   
    #Truncating 
    c = 0
    for p in preG1:
        if p <= trunc:
            preG1[c] = trunc
            #print("Trunc occurs at %.4f" %trunc)
        c += 1    
    
    # Renormalize after truncation
    preG1 = preG1/np.sum(preG1)
    # print("After trunc")
    # print(preG1)
    
    if preG1.size > 11:
        preG1 = preG1[0:11]
        preG1 = preG1/np.sum(preG1)
    else:
        s = preG1.size
        extra = 11 - s
        for add in range(extra):
            np.append(preG1, trunc)
        preG1 = preG1/np.sum(preG1)    
    
    # print("After setting to 11")
    # print(preG1)
    
    
    return preG1

def selfConsistant(g1self, trunc, infinite):
    
    
    g1self = adjustedPoly(g1self, trunc)
    
    #give dx
    g1_prime = derivative(g1self)
    
    
    if g1self.size > 1:
        if g1self.size > 2: 
            g1self[1] = g1self[1] - 1
        elif g1self.size > 1:
            g1self = np.append(g1self, -1)
    else:
        g1self= np.array([0,-1])
    
    
    g1self = np.flip(g1self)

    
    if sum(g1_prime) > 1:
        #print("There is a root below 1")
        all_roots = np.roots(g1self)
        
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
    

def pgfOutbreak(g1Outbreak, trunc, infinite):
    
    
    root = selfConsistant(g1Outbreak, trunc, infinite)
    
    outbreak = 1-root
    
    return root, outbreak


def convertFactor(n, pks):
    
    # if pks.size == 0:
    #     print("EMPTY")
    # elif pks.size >=1:   
    #     if pks.size == 2:
    #         if pks[1] < 0:
    #             print("NEGATIVE")
    
    
    # if pks.size > 1:
    #         if pks.size >= 2: 
    #             pks[1] = pks[1] - 1
               
    #         else:
    #             pks = np.append(pks, -1)
                
    # else:
    #     pks = np.array([0,-1])
   
    
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

def winklerTrunc(n, true_root, true_pk_coeff, noisyCoef, trunc):
    # n = population
    # maxk = max degree
    # true_root = the root from the PGF
    # prime_coeff = coefficients of the derivative function
    # true_pk_coeff = the coefficients of the polynomial G1
    
   
   
    a_i = adjustedPoly(true_pk_coeff, trunc)
    a_plus_error = adjustedPoly(noisyCoef, trunc)
    #Truncating 
    
        
    # a_i = np.array(true_pk_coeff)
    # a_plus_error = np.array(noisyCoef)
    
    # c = 0
    # for p in a_i:
    #     if p <= trunc:
    #         a_i[c] = trunc
    #         #print("Trunc occurs at %.4f" %trunc)
    #     c += 1  
    
    # a_i = a_i/np.sum(a_i)
    # c = 0
    # for ae in a_plus_error:
    #     if ae <= trunc:
    #         a_plus_error[c] = trunc
    #         #print("Trunc occurs at %.4f" %trunc)
    #     c += 1  
    # a_plus_error = a_plus_error/np.sum(a_plus_error)    
        
    
    # making phi a vector 
    phi = np.zeros(a_i.size)
    for m in range(a_i.size):
        phi[m] = true_root**m

    
    
    conversion = 0
    

    if a_i.size == 0:
        return 0
    elif a_i.size == 1 and a_i[0] == 0:
        return 0
    else:
        if a_i.size >= 2: 
            a_i[1] = a_i[1] - 1
            a_plus_error[1] = a_plus_error[1] - 1
            prime_coeff = derivative(a_i)
        else:
            a_i = np.append(a_i, -1)
            a_plus_error = np.append(a_plus_error, -1)
            prime_coeff = derivative(a_i)
        
        
        a_i = a_i[a_i != 0] #takes out the zeros for the Winkler computation

        aInverse = np.true_divide(1, a_i) # This is correct for the l1norm
        
        prime_coeff = np.flip(prime_coeff)
        prime = np.poly1d(prime_coeff)
        
        conversion = convertFactor(n, a_plus_error)
   
        
        
        
        if prime(true_root) == 0:
            winklerMeasure = 0
        else: 
            winklerMeasure = (1/np.abs(prime(true_root))) * np.linalg.norm(phi, 2)/(true_root * np.linalg.norm(aInverse, 1))*conversion
    
    
    return winklerMeasure

def  winkler(n, true_root, true_pk_coeff, noisyCoef):
    # n = population
    # maxk = max degree
    # true_root = the root from the PGF
    # prime_coeff = coefficients of the derivative function
    # true_pk_coeff = the coefficients of the polynomial G1
    
    a_i = adjustedPoly(true_pk_coeff, 0)
    a_plus_error = np.array(noisyCoef)
    
    # making phi a vector 
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
            a_i = np.array([0.001,-1])
            prime_coeff = np.array([-1])    
        
          #a_i = a_i[a_i != 0] #takes out the zeros for the Winkler computation

          aInverse = np.true_divide(1,a_i) # This is correct for the l1norm
        
          prime_coeff = np.flip(prime_coeff)
          prime = np.poly1d(prime_coeff)
        
          conversion = convertFactor(n, a_plus_error)
   
        
        
          if prime(true_root) == 0:
              winklerMeasure= 0
          else: 
              winklerMeasure = (1/np.abs(prime(true_root))) * np.linalg.norm(phi, 2)/(true_root * np.linalg.norm(aInverse, 1))*conversion
    
    #equation 16
    #winklerMeasure = (sigma*np.sqrt(2)/(np.sqrt(math.pi)*true_root))*(1/np.abs(prime(true_root)))*( np.linalg.norm(phi, 2))
    # winklerMeasure = 0
    return winklerMeasure


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
            
            thresh = 1/10000
            
           #Solving for the pgf
            root, outbreak = pgfOutbreak(g1, thresh, True)
            
            U[n,j] = root
            S[n,j] = outbreak
            
            
            #compSens[n,j] = winklerTrunc(10000, U[0,0], g1True, g1, thresh)
            compSens[n,j] = winkler(10000, U[0,0], g1True, g1)
            
    
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
        
        r = 1.5+rh*b # Value for R0 for each iteration
        
        for c in range(1,dim-1):
            
            kLoop =  0.1 + kh*c # Value for k for each iteration
            
            
            
            avgCompSens = mainWithTrueCoef(r, kLoop, col, numSD, inc, maxk)
            
            r0_values[counter] = r
            k_values[counter] = kLoop
            compWinkler[counter] = np.sum(avgCompSens)/len(avgCompSens)
            counter +=1
            
    df_winkler = pd.DataFrame({'R0': r0_values, 'k':k_values, 'Winkler':compWinkler}) 
    return df_winkler        
            
          
#%%%    Running heatmap     
        
            
dim = 15
col = 10
numSD = 3
inc = 0.01
maxk = 50

df_Heatmap = heatmap(dim, col, numSD, inc, maxk)


data_2_plot = df_Heatmap.pivot(index = 'k', columns = 'R0', values = 'Winkler' )
sns.heatmap(data_2_plot, cmap = "Greens")



#%%%% Second heat map with single sigma


def mainWithTrueCoefSigma(r0, k, col, sigma, maxk):
    
    U = np.zeros((col))
    S = np.zeros((col))
    compSens = np.zeros((col))
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
        
        thresh = 1/10000
        
       #Solving for the pgf
        root, outbreak = pgfOutbreak(g1, thresh, True)
        
        U[n] = root
        S[n] = outbreak
        
        
        #compSens[n] = winklerTrunc(10000, U[0], g1True, g1, thresh)
        compSens[n] = winkler(10000, U[0], g1True, g1)
        
    
    # for b in range(col):
    #     avgComp[b] = np.sum(compSens[b,:])/numSD  
        
    return compSens

def heatmapSigma(dim, col, sigma, maxk):
    
    rh = 2/dim
    kh = .15/dim
    
    M = (dim-2)**2
    
    r0_values = np.zeros(M)
    k_values = np.zeros(M)
    compWinkler = np.zeros(M)
    
    counter = 0
    
    for b in range(1,dim-1):
        
        r = 1.5+rh*b # Value for R0 for each iteration
        
        for c in range(1,dim-1):
            
            kLoop =  0.1 + kh*c # Value for k for each iteration
            
            
            
            CompSens = mainWithTrueCoefSigma(r, kLoop, col, sigma, maxk)
            
            r0_values[counter] = r
            k_values[counter] = kLoop
            compWinkler[counter] = np.sum(CompSens)/len(CompSens)
            counter +=1
            
    df_winkler = pd.DataFrame({'R0': r0_values, 'k':k_values, 'Winkler':compWinkler}) 
    return df_winkler        
            
          
#%%%    Running heatmap with constant sigma
        
            
dim = 15
col = 10
sigma = 0.2
maxk = 50

df_Heatmap = heatmapSigma(dim, col, sigma, maxk)


data_2_plot = df_Heatmap.pivot(index = 'k', columns = 'R0', values = 'Winkler' )
sns.heatmap(data_2_plot, cmap = "Greens")