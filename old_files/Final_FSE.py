#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:10:39 2022

@author: mariahboudreau


Final FSE Code with expected U for error bars 
"""

#%%%%%
    
import numpy as np
import pandas as pd
import scipy
import math
from scipy import stats
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter


def Union(fir, sec): #k = nodes dealt with ## v is the current node
    if type(fir) is int:
        fir = [fir]
    if type(sec) is int:
        sec = [sec]
    c = Counter(fir + sec)
    final = list(c.keys())
    return final

def setDiff(fir, sec):
    if type(fir) is int:
        fir = [fir]
    if type(sec) is int:
        sec = [sec]
    
    diff = []
    for item in fir:
        if item not in sec:
            diff.append(item)
    
    diff.sort()
    return diff


def plottingInd(n, deg, pgfSimOutbreak, giantComps, winklerMeasure, true_outbreak):
    
    fig2 = plt.figure(figsize=(10,7))

    plt.axis([-.1,1,-0.1,1])
    plt.title("Erdos-Renyi G(%d, <k>)" %(n))
    plt.xlabel("Predicted from PGF")
    plt.ylabel("Measured Giant Component in Simulations")
    colors = ["red", "blue" , "green", "orange"]
    starColors = ["pink", "cyan", "olive", "yellow"]
    labels = ["k = 0.5", "k = 1.0", "k = 1.5", "k = 2.0"]
    starLabels = ["Predicted Outbreak, k = 0.5", "Predicted Outbreak, k = 1.0", "Predicted Outbreak, k = 1.5", "Predicted Outbreak, k = 2.0"]
    
    ncolor = 0
    for c in range(0,len(pgfSimOutbreak)):
        plt.errorbar(true_outbreak[c], true_outbreak[c], ecolor = starColors[ncolor], color =starColors[ncolor], label = starLabels[ncolor], marker = '*', markersize =20)
        plt.errorbar(pgfSimOutbreak[c,:],giantComps[c,:], xerr= winklerMeasure[c,:], color=colors[ncolor], ecolor = 'grey', label=labels[ncolor], fmt = "o", alpha = 0.5)
       
        ncolor += 1
    
    line = np.linspace(0,1)
    plt.plot(line, line, color = 'black')
    plt.legend()
    plt.show()  
    
    

   
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
    
    

def pgfOutbreak(g1Outbreak):
    
    
    root = selfConsistant(g1Outbreak)
    
    outbreak = 1-root
    
    return root, outbreak


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


def prepLogFunction(coeffs):
    small = min(np.abs(coeffs))
    
    coeffs= np.delete(coeffs , np.where(coeffs == small))
    
    coeffs_fix = np.true_divide(small, coeffs)
    
    return coeffs_fix, small

def definingSigma(degrees, true_degs, size, maxk): # could use analytical value or use standard length. 
    errors = np.empty((maxk-1))
    
    # pad the degrees array to include 0s up to 20 
    if len(degrees) < 20:
        N = 20 - len(degrees)
        degrees = np.pad(degrees, (0, N), 'constant')
    
    for j in range(maxk-1):
        errors[j] = np.abs(degrees[j] - true_degs[j])
    
    mean = np.mean(errors)
    
    temp = np.empty((len(errors)))
    for i in range(len(errors)):
        temp[i] = (errors[i]-mean)**2
    
    sample_var = temp/size
    
    return np.sqrt(sample_var[0])

def expectationOfU_FSE(a_i_coeff, true_coeff, true_root, pop, maxk, simSD):
    
    if simSD:
        sigma = definingSigma(a_i_coeff, true_coeff, pop, maxk)
    else: 
        temp = 2*(true_coeff - true_coeff**2)
        temp = temp/(pop*math.pi)
        sigma = np.sum(np.sqrt(temp))
        
    #print("Sigma analytical: %.2f with pop of %d" %(sigma, pop))
    
    a_i, prime_coeff = prepSelfconsistent(a_i_coeff)
    
    
    # making phi a vector the same size as the coefficients
    phi = np.zeros(a_i.size)
    for k in range(a_i.size):
        phi[k] = true_root**k

    
    
    prime_coeff = np.flip(prime_coeff)
    prime = np.poly1d(prime_coeff)
    
    
    
    
    if sigma == 0:
        value = 0
    else:
        # value = -np.log(true_root) - np.log(np.abs(prime(true_root))) + np.log(np.sqrt(2)) + np.log(sigma) + np.log(np.linalg.norm(phi,2)) - np.log(np.sqrt(math.pi))
        value = 1/(true_root*np.abs(prime(true_root)))*(np.sqrt(2)*sigma*np.linalg.norm(phi,2))/(np.sqrt(math.pi))
    
    
    return value, sigma, prime(true_root), np.linalg.norm(phi,2), true_root


def prepLogFunction(coeffs):
    small = min(np.abs(coeffs))
    
    coeffs= np.delete(coeffs , np.where(coeffs == small))
    
    coeffs_fix = np.true_divide(small, coeffs)
    
    return coeffs_fix, small

def expectationOfCoeff(a_i_coeff, sigma):
    
    a_i, prime_coeff = prepSelfconsistent(a_i_coeff)
    
    aInverse, smallCoef = prepLogFunction(a_i)
    
    bottomValue = np.log(sigma) + np.log(np.sqrt(2)) - np.log(np.sqrt(math.pi)) + np.log(np.linalg.norm(aInverse,1))
    
    return np.exp(bottomValue)

#%%%%%
def runFSE(excessDegree, numTrials, exp, maxk):

    
    for N in range(1, exp+1):    
        
        
        
        outbreakSize = np.zeros((len(excessDegree), numTrials)) #Creating matrices 
        giantComps = np.zeros((len(excessDegree), numTrials))
        pgfSimOutbreak = np.zeros((len(excessDegree), numTrials))
        true_outbreak = np.zeros(len(excessDegree))
        winklerMeasure = np.zeros((len(excessDegree), numTrials))
        polyLengths = np.zeros((len(excessDegree)*numTrials + len(excessDegree)))
        plotWink = np.zeros((len(excessDegree), 4))
        plotWinkInf = np.zeros((len(excessDegree), 4))
        
        
        for deg, count in zip(excessDegree, range(len(excessDegree))):
            
                
            n = 10**(N) # population size
                
            prob = deg/(n-1) # probability of connection 
           # print('Degree: %.2f' %deg)
            
            # Calculating the true outbreak from a ER graph
            true_p_k_infinite_G0 = ERCoeff(maxk, prob, n)
            
            
            #Derivative
            true_prime_infinite_G1 = derivative(true_p_k_infinite_G0)
            
            if sum(true_prime_infinite_G1) != 0:
                true_prime_infinite_G1 = true_prime_infinite_G1/sum(true_prime_infinite_G1)
           
            true_root, true_outbreak[count]= pgfOutbreak(true_prime_infinite_G1)
           
    
            converts = np.zeros((numTrials))
            
            
            for t in range(0, numTrials):
                
                
                p_k_sim_G0, G = ER_FSE_Sim(n, prob)
                
                    
                p_k_sim_G1 = derivative(p_k_sim_G0) # G1 for the simulations
                
                
                if p_k_sim_G1.size == 0:
                    p_k_sim_G1 = np.array([0,0])
                elif p_k_sim_G1.size >= 1:
                    if np.sum(p_k_sim_G1) != 0:
                        p_k_sim_G1 = p_k_sim_G1/np.sum(p_k_sim_G1)
                    
                
                    
                 
            
                root, outbreak = pgfOutbreak(p_k_sim_G1) 
                
                pgfSimOutbreak[count, t] = outbreak
                
                
                
                # Find all neighbors
                neighbors = []
                for i in range(0,n):
                    temp = list(G.neighbors(i))
                    neighbors.append(temp)
                
                
                x = np.zeros(n) # index which connect comp a node is a part of
                numComp = 0 # Number of components tat the start
                k = [] # Nodes dealt with
                
                # Loop over nodes to see which component they are in
                for j in range(0,n):
                    if x[j] == 0:
                        numComp = numComp + 1
                        v = []
                        v.append(j) # set of nodes in the current component
                        while v != []:
                            x[v[0]] = numComp     # assigning node to its component
                            k = Union(k,v[0])              # Combines current node to the comp
                            p = setDiff(neighbors[v[0]], k)# Finds the neighbors not in comp yet
                            v = setDiff(v, v[0])           # Gets ride of node in stack
                            v = Union(v,p) # preps nodes to be added to component
                            
                            
                            
                            
                # Figure out size of components    
                
                c = np.zeros(int(max(x))) # Number of components  
                
                lengths = Counter(x).values() # Sizes of components
                
                outbreakSize[count, t] = max(lengths) # Size of largest component 
                
                giantComps[count, t] = outbreakSize[count,t]/n# Percentage of population
                
                
                
                winklerMeasure[count, t]=  expectationOfU_FSE(p_k_sim_G1, true_prime_infinite_G1, true_root, n, maxk, False)
                                                               
                    
                    
                        
           
        plottingInd(n, deg, pgfSimOutbreak, giantComps,  winklerMeasure, true_outbreak)
        #print(winklerMeasure)
        
        
        
        
excessDegree = [.5, 1, 1.5, 2]
numTrials = 100
exponent = 3
maxk = 20
runFSE(excessDegree, numTrials, exponent, maxk)


#%%%%% bifurcation diagram 


   
import numpy as np
import pandas as pd
import scipy
import math
from scipy import stats
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

def plotting_S_and_k():
    avgK = np.linspace(0,3, num=31)
    numTrials = 100
    exp = 3
    maxk = 20
    #This will be the plot for the threshold 
    for N in range(1, exp+1):    
        outbreakSize = np.zeros((len(avgK), numTrials)) #Creating matrices 
        giantComps = np.zeros((len(avgK), numTrials))
        pgfSimOutbreak = np.zeros((len(avgK), numTrials))
        true_outbreak = np.zeros(len(avgK))
        winklerMeasure = np.zeros((len(avgK), numTrials))
        sigmas = np.zeros((len(avgK), numTrials))
        primes = np.zeros((len(avgK), numTrials))
        norms = np.zeros((len(avgK), numTrials))
        truRoot = np.zeros((len(avgK), numTrials))
        
        count = 0
        leng = 0
        for deg in avgK:
            
            n = 10**(N) # population size
                
            prob = deg/(n-1) # probability of connection 
            
            # Calculating the true outbreak from a ER graph
            true_p_k_infinite_G0 = ERCoeff(maxk, prob, n)
            
            
            #Derivative
            true_prime_infinite_G1 = derivative(true_p_k_infinite_G0)
            
            if sum(true_prime_infinite_G1) != 0:
                true_prime_infinite_G1 = true_prime_infinite_G1/sum(true_prime_infinite_G1)
           
            true_root, true_outbreak[count] = pgfOutbreak(true_prime_infinite_G1)
            
            
            
            
            
            for t in range(0, numTrials):
                
                p_k_sim_G0, G = ER_FSE_Sim(n, prob)
                
                    
                p_k_sim_G1 = derivative(p_k_sim_G0) # G1 for the simulations
                
                
                if p_k_sim_G1.size == 0:
                    p_k_sim_G1 = np.array([0,0])
                elif p_k_sim_G1.size >= 1:
                    if np.sum(p_k_sim_G1) != 0:
                        p_k_sim_G1 = p_k_sim_G1/np.sum(p_k_sim_G1)
                    
                
                    
                 
            
                root, outbreak = pgfOutbreak(p_k_sim_G1)
                
                pgfSimOutbreak[count, t] = outbreak
                
                
                
                # Find all neighbors
                neighbors = []
                for i in range(0,n):
                    temp = list(G.neighbors(i))
                    neighbors.append(temp)
                
                
                x = np.zeros(n) # index which connect comp a node is a part of
                numComp = 0 # Number of components tat the start
                k = [] # Nodes dealt with
                
                # Loop over nodes to see which component they are in
                for j in range(0,n):
                    if x[j] == 0:
                        numComp = numComp + 1
                        v = []
                        v.append(j) # set of nodes in the current component
                        while v != []:
                            x[v[0]] = numComp     # assigning node to its component
                            k = Union(k,v[0])              # Combines current node to the comp
                            p = setDiff(neighbors[v[0]], k)# Finds the neighbors not in comp yet
                            v = setDiff(v, v[0])           # Gets ride of node in stack
                            v = Union(v,p) # preps nodes to be added to component
                            
                            
                            
                            
                # Figure out size of components    
                
                c = np.zeros(int(max(x))) # Number of components  
                
                lengths = Counter(x).values() # Sizes of components
                
                outbreakSize[count, t] = max(lengths) # Size of largest component 
                
                giantComps[count, t] = outbreakSize[count,t]/n# Percentage of population
                
                
                
                #winklerMeasure[count, t], sigmas[count, t], primes[count, t], norms[count, t], truRoot[count,t] =  expectationOfU_FSE(p_k_sim_G1, true_prime_infinite_G1, true_root, n, maxk, False, deg)
                leng+=1
                #(1/np.abs(prime(true_root))) * np.abs(np.linalg.norm([phi(true_root)], 2))/(true_root * np.linalg.norm([denomP], 1))* np.sqrt(2/(math.pi*n)*(denomP))
            
            
            
            count += 1
        
        fig2 = plt.figure(figsize=(10,7))
        
        
        
        #winklerMeasure[winklerMeasure == np.inf] = 0 
            
        avgGC = np.zeros((len(avgK)))    
        for row in range(len(avgK)):
            avgGC[row] = sum(giantComps[row][:])/numTrials 
            
          
        # avgWinkler = np.zeros((len(avgK)))    
        # for row in range(len(avgK)):
        #     avgWinkler[row] = sum(winklerMeasure[row][:])/numTrials 
            
       
        #avgWinklerInf = sum(winklerMeasureInfinite)/len(avgK)

       #print(winklerMeasure)
        print(sigmas)
        print(norms)
        print(primes)
        print(truRoot)
        
        plt.axis([0,3,0,1])
        plt.title("Population N = %d, Outbreak Threshold" %(n))
        plt.xlabel("Average secondary degree")
        plt.ylabel("S")
        
        #print(winklerMeasure)
        
        plt.plot(avgK, true_outbreak, color = 'black')
        #plt.errorbar(avgK, true_outbreak, yerr = avgWinkler, color = 'black', ecolor = 'black')
        plt.scatter(avgK, avgGC, color='black')
        
        for col in range(numTrials):
            plt.scatter(avgK, giantComps[:,col], alpha= 0.05, color = 'blue')
        plt.show()
        
        
        
        
plotting_S_and_k()


# Clean this code up


# %%
