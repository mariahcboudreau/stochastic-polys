#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mariah Boudreau
Finite Size Effects - PGF Sensitivity Analysis
"""




#%% Plotting a different graph, need to change the for loop ordering
    
    
    
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


def plottingInd(n, deg, pgfSimOutbreak, giantComps, winklerMeasureInfinite, winklerMeasure, true_outbreak):
    
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
        plt.errorbar(true_outbreak[c], true_outbreak[c], yerr = winklerMeasureInfinite[c], ecolor = starColors[ncolor], color =starColors[ncolor], label = starLabels[ncolor], marker = '*', markersize =20)
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



#This function will take in the PGF and adjust it into the polynomial we want to use to solve for the root/use for Winkler measure
# def adjustedPoly(pgfCoeff, trunc): 
#     preG1 = np.array(pgfCoeff)
    
   
#     #Truncating 
#     c = 0
#     for p in preG1:
#         if p <= trunc:
#             preG1[c] = trunc
#         c += 1    
    
#     # Renormalize after truncation
        
#     if np.sum(preG1) == 0:
#         preG1 = np.zeros(1)
#     else:
#         preG1 = preG1/np.sum(preG1)
   
    
#     if preG1.size > 11:
#         preG1 = preG1[0:11]
#         preG1 = preG1/np.sum(preG1)
#     else:
#         s = preG1.size
#         extra = 11 - s
#         for add in range(extra):
#             np.append(preG1, trunc)
#         preG1 = preG1/np.sum(preG1)    
    
#     # checkAvgK = derivative(preG1)
#     # print("Average K:")
#     # print(np.sum(checkAvgK))
    
#     return preG1

def prepLogFunction(coeffs):
   
    
    small = min(np.abs(coeffs)) 
    
    
    coeffs= np.delete(coeffs , np.where(coeffs == small))
    
    coeffs_fix = np.true_divide(small, coeffs)
    
    return coeffs_fix, small

def selfConsistant(g1, trunc, infinite):
    
    # if infinite == False:
    #     g1 = adjustedPoly(g1, trunc)
    
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

def winklerTrunc(n, true_root, true_pk_coeff, trunc):
    # n = population
    # maxk = max degree
    # true_root = the root from the PGF
    # prime_coeff = coefficients of the derivative function
    # true_pk_coeff = the coefficients of the polynomial G1
    
   
  
    a_i = np.array(true_pk_coeff)
       
        
    
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
                #a_plus_error[1] = a_plus_error[1] - 1
                prime_coeff = derivative(a_i)
            else:
                a_i = np.append(a_i, -1)
                #a_plus_error = np.append(a_plus_error, -1)
                prime_coeff = derivative(a_i)
         else:
            a_i = np.array([0,-1])
            #a_plus_error = np.array([0,-1])
            prime_coeff = np.array([-1])    
        
         a_i = a_i[a_i != 0] #takes out the zeros for the Winkler computation
         
         if a_i.size == 0:
             winklerMeasure = 0 
         else:
         # aInverse = np.true_divide(1, a_i) # This is correct for the l1norm
        
         # prime_coeff = np.flip(prime_coeff)
         # prime = np.poly1d(prime_coeff)
        
         # conversion = convertFactor(n, a_i)
        
             aInverse, smallCoef = prepLogFunction(a_i)
            
             prime_coeff = np.flip(prime_coeff)
             prime = np.poly1d(prime_coeff)
            
             conversion = convertFactor(n, a_i)
       
            
            
            
             if prime(true_root) == 0:
                 winklerMeasure = 0
             elif np.linalg.norm(aInverse,1) == 0:
                 winklerMeasure = 0
             else:
                # winklerMeasure = np.log((1/np.abs(prime(true_root)))* np.linalg.norm(phi, 2)/(true_root * np.linalg.norm((aInverse - smallCoef), 1))*conversion)
                 winklerMeasure = -np.log(np.abs(prime(true_root))) + np.log(np.linalg.norm(phi,2)) - np.log(true_root) - np.log(np.linalg.norm(aInverse,1)) + np.log(smallCoef) + np.log(conversion)
        
        
         # if prime(true_root) == 0:
         #     winklerMeasure = 0
         # else: 
         #     winklerMeasure = (1/np.abs(prime(true_root))) * np.linalg.norm(phi, 2)/(true_root * np.linalg.norm(aInverse, 1))*conversion
    
    
    return np.exp(winklerMeasure)

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
    
    return root, outbreak, g1Outbreak


def runFSE(excessDegree, numTrials, exp, maxk):

    
    for N in range(1, exp+1):    
        
        
        
        outbreakSize = np.zeros((len(excessDegree), numTrials)) #Creating matrices 
        giantComps = np.zeros((len(excessDegree), numTrials))
        pgfSimOutbreak = np.zeros((len(excessDegree), numTrials))
        true_outbreak = np.zeros(len(excessDegree))
        winklerMeasure = np.zeros((len(excessDegree), numTrials))
        winklerMeasureInfinite = np.zeros(len(excessDegree))
        polyLengths = np.zeros((len(excessDegree)*numTrials + len(excessDegree)))
        plotWink = np.zeros((len(excessDegree), 4))
        plotWinkInf = np.zeros((len(excessDegree), 4))
        
        count = 0
        for deg in excessDegree:
            
            n = 10**(N) # population size
                
            prob = deg/(n-1) # probability of connection 
           # print('Degree: %.2f' %deg)
            
            # Calculating the true outbreak from a ER graph
            true_p_k_infinite_G0 = ERCoeff(maxk, prob, n)
            
            
            #Derivative
            true_prime_infinite_G1 = derivative(true_p_k_infinite_G0)
            
            if sum(true_prime_infinite_G1) != 0:
                true_prime_infinite_G1 = true_prime_infinite_G1/sum(true_prime_infinite_G1)
           
            true_root, true_outbreak[count] , true_prime_infinite_G1= pgfOutbreak(true_prime_infinite_G1)
            
          
            
            #winkler
             
            winklerMeasureInfinite[count] = winklerTrunc(n, true_root, true_prime_infinite_G1, 0)
            # Need to count the number of polynomials 
            
            
                   
            
        
            winkler100s = np.zeros((4,numTrials))
            winklerInf100s = np.zeros((4))
            converts = np.zeros((numTrials))
            m=0
            for cutoff in [0, 0.001, 0.01, 0.1]:
                for t in range(0, numTrials):
                    
                    
                    p_k_sim_G0, G = ER_FSE_Sim(n, prob)
                    
                        
                    p_k_sim_G1 = derivative(p_k_sim_G0) # G1 for the simulations
                    
                    
                    if p_k_sim_G1.size == 0:
                        p_k_sim_G1 = np.array([0,0])
                    elif p_k_sim_G1.size >= 1:
                        if np.sum(p_k_sim_G1) != 0:
                            p_k_sim_G1 = p_k_sim_G1/np.sum(p_k_sim_G1)
                        
                    
                        
                     
                
                    root, outbreak, p_k_sim_G1 = pgfOutbreak(p_k_sim_G1)
                    
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
                    
                    
                    thresh = 1/n
                    
                    winklerMeasure[count, t]=  winklerTrunc(n, true_root, p_k_sim_G1, thresh)
                    
                    
                    
                    winkler100s[m, t] =  winklerTrunc(n, true_root, p_k_sim_G1, cutoff)
                        
                    winklerInf100s[m] = winklerTrunc(n, true_root, true_prime_infinite_G1, cutoff)
                m+=1
                        
                    #(1/np.abs(prime(true_root))) * np.abs(np.linalg.norm([phi(true_root)], 2))/(true_root * np.linalg.norm([denomP], 1))* np.sqrt(2/(math.pi*n)*(denomP))
                
            
            #print(winkler100s) #check this out
            winkler100sAvg = np.zeros((4))
            for h in range(len(winkler100s)):
                winkler100sAvg[h] = np.sum(winkler100s[h])/numTrials
            
            #print(winkler100sAvg)
            plotWink[count,:] = winkler100sAvg
            plotWinkInf[count,:] = winklerInf100s
            
            count += 1
            #avgConverts = np.sum(converts)/len(converts)
            # print("Average Conversion for n = %d: %.5f"%(n,avgConverts))
        # labels = ["k = 0.5", "k = 1.0", "k = 1.5", "k = 2.0"]  
        # colors = ["red", "blue" , "green", "orange"]
        # for b in range(len(labels)):
        #     plt.plot(np.arange(len(plotWinkInf[b])),plotWinkInf[b], marker = "o",label = labels[b], color = colors[b])
        #     X = np.arange(0.2,4.2,step = 1)
        #     plt.plot(X, plotWink[b], marker = '*', label = labels[b], color = colors[b], linestyle='dashed')
        # plt.ylim(-0.1,1.5)
        # plt.xticks(np.arange(0,4,step = 1), ["0", "0.001", "0.01" ,"0.1"])
        # plt.xlabel("Truncation")
        # plt.ylabel("Winkler Measure")
        # plt.title("Varying Truncation of Polynomial for Winkler Measure for n = %d" %n)
        
        # plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol= 2)
        # plt.show()    
        plottingInd(n,deg, pgfSimOutbreak, giantComps, winklerMeasureInfinite,  winklerMeasure, true_outbreak)
        #print(infConverts)
        #print(winklerMeasureInfinite)
        # print(winklerMeasure)
        
            
        
         
               

  
    
excessDegree = [.5, 1, 1.5, 2]
numTrials = 100
exponent = 3
maxk = 20
runFSE(excessDegree, numTrials, exponent, maxk)


#%%

    
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
        winklerMeasureInfinite = np.zeros(len(avgK))
        polyLengths = np.zeros((len(avgK)*numTrials + len(avgK)))
        
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
           
            true_root, true_outbreak[count], true_prime_infinite_G1 = pgfOutbreak(true_prime_infinite_G1)
            
            thresh = 1/n
            
            # if (deg >= 2) and (n > 10):
            #     print("Stop and check")
            
            winklerMeasureInfinite[count] = winklerTrunc(n, true_root, true_prime_infinite_G1, thresh)
           
            
            for t in range(0, numTrials):
                
                p_k_sim_G0, G = ER_FSE_Sim(n, prob)
                
                    
                p_k_sim_G1 = derivative(p_k_sim_G0) # G1 for the simulations
                
                
                if p_k_sim_G1.size == 0:
                    p_k_sim_G1 = np.array([0,0])
                elif p_k_sim_G1.size >= 1:
                    if np.sum(p_k_sim_G1) != 0:
                        p_k_sim_G1 = p_k_sim_G1/np.sum(p_k_sim_G1)
                    
                
                    
                 
            
                root, outbreak, p_k_sim_G1 = pgfOutbreak(p_k_sim_G1)
                
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
                
                
                
                winklerMeasure[count, t]=  winklerTrunc(n, true_root, p_k_sim_G1, thresh)
                leng+=1
                #(1/np.abs(prime(true_root))) * np.abs(np.linalg.norm([phi(true_root)], 2))/(true_root * np.linalg.norm([denomP], 1))* np.sqrt(2/(math.pi*n)*(denomP))
            
            
            
            count += 1
        
        fig2 = plt.figure(figsize=(10,7))
        
        
     
            
            
        avgGC = np.zeros((len(avgK)))    
        for row in range(len(avgK)):
            avgGC[row] = sum(giantComps[row][:])/numTrials 
            
        avgWinkler = np.zeros((len(avgK)))    
        for row in range(len(avgK)):
            avgWinkler[row] = sum(winklerMeasure[row][:])/numTrials 
            
        #avgWinklerInf = sum(winklerMeasureInfinite)/len(avgK)

        
        plt.axis([0,3,0,1])
        plt.title("Population N = %d, Outbreak Threshold" %(n))
        plt.xlabel("<k>")
        plt.ylabel("S")
        
        print(winklerMeasureInfinite)
        
        plt.plot(avgK, true_outbreak, color = 'black')
        plt.errorbar(avgK, true_outbreak, yerr = winklerMeasureInfinite, color = 'black', ecolor = 'black')
        plt.scatter(avgK, avgGC)
        #plt.errorbar(avgK, avgGC, yerr= avgWinkler,  ecolor = 'grey', fmt = 'o')
        plt.show()
        
        
        
        
plotting_S_and_k()


# Clean this code up
