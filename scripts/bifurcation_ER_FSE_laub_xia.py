#####
#
# Bifurcation diagram
#
#####

############################# IMPORTS ############################


from stochastic_pgfs.laub_xia_algo import polynomial_roots, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.sim import *
import networkx as nx
from collections import Counter
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
date = datetime.today().strftime('%m-%d-%Y')

########################## AUXILARY FUNCTIONS #########################

def ERCoeff(maxk, prob, n):
    binomCoeff = []
    for k in range(maxk):
        binomCoeff.append(scipy.special.binom(n-1, k)*prob**(k)*(1-prob)**(n-1-k))
    
    return binomCoeff


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

def giant_comps(G, n):

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

    return max(lengths), max(lengths)/n

########################## THRESHOLD FIGURE ###########################
def plotting_S_and_k():
    avgK = np.linspace(0,3, num=31)
    numTrials = 100
    exp = 3
    maxk = 20
 
    outbreakSize = np.zeros((len(avgK), numTrials)) #Creating matrices 
    giantComps = np.zeros((len(avgK), numTrials))
    pgfSimOutbreak = np.zeros((len(avgK), numTrials))
    true_outbreak = np.zeros(len(avgK))
    true_outbreak_origin = np.zeros(len(avgK))
    condition_nums = np.zeros(len(avgK))
    diffs = []
    true_Root = np.zeros(len(avgK))
    pgfSimRoot = np.zeros((len(avgK), numTrials))
    
    n = 1000 # population size
    
    count = 0
  
    for deg in avgK:
        
            
        prob = deg/(n-1) # probability of connection 
        
        # Calculating the true outbreak from a ER graph
        true_p_k_infinite_G0 = ERCoeff(maxk, prob, n)
        
        # #Derivative
        true_prime_infinite_G1 = derivative(true_p_k_infinite_G0)
        # true_prime_infinite_G1 = ERCoeff(maxk, prob, n)

        if sum(true_prime_infinite_G1) != 0:
            true_prime_infinite_G1 = true_prime_infinite_G1/sum(true_prime_infinite_G1)

        true_root, true_outbreak_origin[count] = pgfOutbreak(true_prime_infinite_G1)


        if np.sum(derivative(true_prime_infinite_G1)) > 1:
            my_pgf_coef = make_G_u_minus_u(true_prime_infinite_G1)
            all_og_roots = polynomial_roots(np.flip(my_pgf_coef))
            all_conditions = np.array([True] * len(all_og_roots))
                
            all_conditions = np.logical_and.reduce(
                [cond(all_og_roots) for cond in [is_real, in_bounds]]
            )
            if len(all_og_roots[all_conditions]) > 1:
                true_Root[count] = np.max(all_og_roots[all_conditions])
            elif len(all_og_roots[all_conditions]) == 0:
                true_Root[count] = 1
            else:
                true_Root[count] = all_og_roots[all_conditions][0]
        else:
            true_Root[count] = 1
            my_pgf_coef = make_G_u_minus_u(true_prime_infinite_G1)
            all_og_roots = polynomial_roots(np.flip(my_pgf_coef))

        # my_pgf_coef = make_G_u_minus_u(true_prime_infinite_G1)
        # all_og_roots = polynomial_roots(my_pgf_coef)

        # all_conditions = np.array([True] * len(all_og_roots))
       
        # all_conditions = np.logical_and.reduce(
        #     [cond(all_og_roots) for cond in [is_real, in_bounds]]
        # )
        # if len(all_og_roots[all_conditions]) != 0:
        #     true_Root[count] = np.max(all_og_roots[all_conditions])
        # else:
        #     true_Root[count] = 1


        diffs.append(l_x_algo(true_prime_infinite_G1, K=1000, conditions=[is_real, in_bounds], is_pgf = True, perturbation_type="additive", bifurcation=True))

        true_outbreak[count] = 1-true_Root[count]


        for t in range(0, numTrials):
            
            p_k_sim_G0, G = ER_FSE_Sim(n, prob)
            
            p_k_sim_G1 = derivative(p_k_sim_G0) # G1 for the simulations

            p_k_sim_G1 = np.array(p_k_sim_G1)
            
            if p_k_sim_G1.size == 0:
                p_k_sim_G1 = np.array([0,0])
            elif p_k_sim_G1.size >= 1:
                if np.sum(p_k_sim_G1) != 0:
                    p_k_sim_G1 = p_k_sim_G1/np.sum(p_k_sim_G1)

            # pgfSimRoot[count, t], pgfSimOutbreak[count, t] = solve_self_consistent_roots(p_k_sim_G1)
            
            

            outbreakSize[count, t], giantComps[count, t] = giant_comps(G, n)

            
        count += 1
    
    
    
    
        
    avgGC = np.zeros((len(avgK)))    
    for row in range(len(avgK)):
        avgGC[row] = sum(giantComps[row][:])/numTrials 
        
        
    # fig2 = plt.figure(figsize=(10,7))
########################### PLOTTING ###############################
    from matplotlib.lines import Line2D

    plt.axis([0,3,0,1])
    plt.title("Population N = %d, Outbreak Threshold" %(n))
    plt.xlabel("Average secondary degree")
    plt.ylabel("S")

    #print(winklerMeasure)
    plt.scatter(avgK, avgGC, color='red', label = "Average ER Giant Component")
    plt.plot(avgK, true_outbreak, color = 'black', label = "True Outbreak")
    #plt.errorbar(avgK, true_outbreak, yerr = condition_nums, color = 'black', ecolor = 'black')
    

    for col in range(numTrials):
        plt.scatter(avgK, giantComps[:,col], alpha= 0.05, color = 'blue')
    handles, labels = plt.gca().get_legend_handles_labels()
    sim_points = Line2D([0], [0], label='', marker = "o", 
         color = "blue", linestyle='')
    handles.extend([sim_points])
    plt.show()
    


########################## TEST ##############################
    
plotting_S_and_k()