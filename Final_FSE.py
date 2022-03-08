#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:10:39 2022

@author: mariahboudreau


Final FSE Code with expected U for error bars 
"""


    
    
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
    
    

def pgfOutbreak(g1Outbreak, trunc, infinite):
    
    
    root = selfConsistant(g1Outbreak, trunc, infinite)
    
    outbreak = 1-root
    
    return root, outbreak

