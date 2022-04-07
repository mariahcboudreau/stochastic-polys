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

from matplotlib import cm
import matplotlib as mpl


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
            
            print(np.sd(g1))
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
    
    #print(slopes(standDevs, sdValues, r0, k))
    #print(standDevs)
    violins(U)
    #ridgePlots(U)    
    #return U[0,0]


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
    
    return expectU

        
#%%% Constructing box plots/ridgeplots
    
r0 = 1.9
k = 0.21
col = 100
numSD = 25
inc = 0.01
maxk = 20
# need to incorporate the noisy values into the measure
UValues = main(r0, k, col, numSD, inc, maxk)



    
r0 = 3.7
k = 0.14
col = 100
numSD = 25
inc = 0.01
maxk = 20
# need to incorporate the noisy values into the measure
UValues = main(r0, k, col, numSD, inc, maxk)






    
#%%% Constructing values for contour plots  (root and 1-root)  
    
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

minus_root_vals = 1-root_vals

df_roots = pd.DataFrame({'R0': r0_vals, 'k':k_vals, 'Roots':root_vals}) 

df_outbreak = pd.DataFrame({'R0': r0_vals, 'k':k_vals, 'Outbreak':minus_root_vals}) 

pivot_root = df_roots.pivot(index='k', columns = 'R0', values = 'Roots')

pivot_outbreak = df_outbreak.pivot(index='k', columns = 'R0', values = 'Outbreak')


#%%%%% Make a contour plot for the delta root values 


sigma = 0.1

r0_vec = np.arange(1, 5, 0.1)
k_vec = np.arange(0.01, 1, 0.01)

r0_vals = np.empty((len(r0_vec)*len(k_vec)))
k_vals = np.empty((len(r0_vec)*len(k_vec)))
delta_root_vals = np.empty((len(r0_vec)*len(k_vec)))

count = 0
for r0 in r0_vec:
    for k in k_vec:
            
        r0_vals[count] = r0
        k_vals[count] = k
        delta_root_vals[count] = solvingForExpectation([r0, k], sigma)
        count += 1

division_vals = delta_root_vals/root_vals
division_out_vals = delta_root_vals/minus_root_vals

df_delta_roots = pd.DataFrame({'R0': r0_vals, 'k':k_vals, 'Expectation':delta_root_vals}) 

df_division_roots = pd.DataFrame({'R0': r0_vals, 'k':k_vals, 'Division':division_vals}) 

df_division_outbreak = pd.DataFrame({'R0': r0_vals, 'k':k_vals, 'Division out':division_out_vals}) 

pivot_division = df_division_roots.pivot(index='k', columns = 'R0', values = 'Division')

pivot_division_out = df_division_outbreak.pivot(index='k', columns = 'R0', values = 'Division out')

pivot_delta = df_delta_roots.pivot(index='k', columns = 'R0', values = 'Expectation')

# #%%%% Constructing the values for the heatmap and the heat map
# mpl.rcParams['text.usetex'] = False
# #mpl.rcParams.update(mpl.rcParamsDefault)
    
# sigma = 0.1

# r0_vec = np.arange(2, 5, 0.1)
# k_vec = np.arange(0.175, 1, 0.01)

# r0_vals = np.empty((len(r0_vec)*len(k_vec)))
# k_vals = np.empty((len(r0_vec)*len(k_vec)))
# expect_root_vals= np.empty((len(r0_vec)*len(k_vec)))

# count = 0
# for r0 in r0_vec:
#     for k in k_vec:
            
#         r0_vals[count] = r0
#         k_vals[count] = k
#         expect_root_vals[count] = solvingForExpectation([r0, k], sigma)
#         count += 1

# df_delta_roots = pd.DataFrame({'avgDegree': r0_vals, 'dispersion':k_vals, 'Expectation':expect_root_vals}) 

# heatmap_plot = df_delta_roots.pivot(index = 'dispersion', columns = 'avgDegree', values = 'Expectation' )


# sns.heatmap(heatmap_plot, cmap = "Greens")
# plt.show()



#%%%% CONTOUR PLOTS




 ###### CONTOUR PLOT FOR PGF ROOT ##########

mpl.rcParams.update(mpl.rcParamsDefault)


fig, axs = plt.subplots(2,2, sharex=True, sharey= True)

####### SEPARATE COLORMAP #########
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(
        norm = norm,
        cmap=cm.viridis
    )
fig.colorbar(sm, ax = axs[0,1], orientation='vertical')
fig.colorbar(sm, ax = axs[1,1], orientation='vertical')

####### CONTOUR PLOT FOR ROOT #########
X = pivot_root.columns.values
Y = pivot_root.index.values
Z = pivot_root.values

x_r,y_r = np.meshgrid(X, Y)

levels = np.arange(0, 1, 0.025)

con_root = axs[0,0].contour(x_r,y_r, Z, cmap=cm.viridis, norm =norm, levels=levels)
axs[0,0].set_title('PGF Root')
axs[0,0].clabel(con_root, inline=1, fontsize=6)
axs[0,0].set_ylabel('k, dispersion parameter')

####### CONTOUR PLOT FOR OUTBREAK #########
X = pivot_outbreak.columns.values
Y = pivot_outbreak.index.values
Z = pivot_outbreak.values

x_o,y_o = np.meshgrid(X, Y)

levels = np.arange(0, 1, 0.025)

con_outbreak = axs[0,1].contour(x_o,y_o, Z, cmap=cm.viridis, norm =norm, levels=levels)
axs[0,1].set_title('Outbreak size')
axs[0,1].clabel(con_outbreak, inline=1, fontsize=6)


###### CONTOUR PLOT FOR DELTA ROOT #######

X = pivot_delta.columns.values
Y = pivot_delta.index.values
Z = pivot_delta.values

x_delta,y_delta = np.meshgrid(X, Y)

levels = np.arange(0, 5, 0.1)

con_delta = axs[1,0].contour(x_delta, y_delta, Z, cmap=cm.viridis, norm =norm, levels=levels)
axs[1,0].set_title('E[$\Delta$ Root]')
axs[1,0].clabel(con_delta, inline=1, fontsize=6)
axs[1,0].set_xlabel('$R_{0}$, average secondary degree')
axs[1,0].set_ylabel('k, dispersion parameter')

###### CONTOUR PLOT FOR DIVISION #######

# X = pivot_division.columns.values
# Y = pivot_division.index.values
# Z = pivot_division.values

# x_div,y_div = np.meshgrid(X, Y)

# levels = np.arange(0, 5, 0.1)

# con_div = axs[1,1].contour(x_div, y_div, Z, cmap=cm.viridis, norm =norm, levels=levels)
# axs[1,1].set_title('E[$\Delta$ Root]/Root')
# #axs[1,1].clabel(con_div, inline=1, fontsize=6)


###### CONTOUR PLOT FOR DIVISION #######

X = pivot_division_out.columns.values
Y = pivot_division_out.index.values
Z = pivot_division_out.values

x_div_o,y_div_o = np.meshgrid(X, Y)

levels = np.arange(0, 5, 0.1)

con_div_o = axs[1,1].contour(x_div_o, y_div_o, Z, cmap=cm.viridis, norm =norm, levels=levels)
axs[1,1].set_title('E[$\Delta$ Root]/Outbreak')
axs[1,1].clabel(con_div_o, inline=1, fontsize=6)
axs[1,1].set_xlabel('$R_{0}$, average secondary degree')


    
    


plt.show()





#%%%% Contour plot for delta root

mpl.rcParams.update(mpl.rcParamsDefault)


fig, ax = plt.subplots()

####### SEPARATE COLORMAP #########
norm = mpl.colors.Normalize(vmin=0, vmax=5)
sm = cm.ScalarMappable(
        norm = norm,
        cmap=cm.viridis
    )
fig.colorbar(sm, ax = ax, orientation='vertical', label='E[$\delta$ Root]')

####### COUNTOUR PLOT #########



plt.clabel(con, inline=1, fontsize=8)

plt.xlabel("$R_{0}$, average secondary degree")
plt.ylabel("k, dispersion parameter")

plt.show()



#%%%% Retrieving the vertices from the contour plots (vertice sets have the contour lines from 0 to 1)

vertice_sets = cs.allsegs


## For 0.325 line
three_two_five_line = []
for j in range(len(vertice_sets[13])):
    for k in range(len(vertice_sets[13][j])):
        three_two_five_line.append(vertice_sets[13][j][k])
    
## For 0.600 line
six_line = []
for j in range(len(vertice_sets[24])):
    for k in range(len(vertice_sets[24][j])):
        six_line.append(vertice_sets[24][j][k])
        

    
## For 0.975 line
nine_seven_five_line = []
for j in range(len(vertice_sets[39])):
    for k in range(len(vertice_sets[39][j])):
        nine_seven_five_line.append(vertice_sets[39][j][k])
        

        
        
#%%% Constructing the expectation of u for specific contour lines 
        
sigma = 0.1
        
three_two_five_line_deltaRoot = np.empty((len(three_two_five_line)))

count = 0
for item in three_two_five_line:
    three_two_five_line_deltaRoot[count] = solvingForExpectation(item, sigma)
    count += 1
  
six_line_deltaRoot = np.empty((len(six_line))) 

count = 0
for item in six_line:
    six_line_deltaRoot[count] = solvingForExpectation(item, sigma)
    count += 1    
    
    
nine_seven_five_line_deltaRoot = np.empty((len(nine_seven_five_line))) 
    
count = 0
for item in nine_seven_five_line:
    nine_seven_five_line_deltaRoot[count] = solvingForExpectation(item, sigma)
    count += 1
    
x325 = np.arange(len(three_two_five_line))
x6 = np.arange(len(six_line))
x975 = np.arange(len(nine_seven_five_line))

#plt.scatter(x325, three_two_five_line_deltaRoot, label='Root = 0.325') 
# Starts at (4.9, 0.6410) to (3.11, 0.99)

#plt.scatter(x6, six_line_deltaRoot, label='Root = 0.6')
# Starts at (4.9, 0.2792) to (1.67, 0.99)

plt.scatter(x975, nine_seven_five_line_deltaRoot, label='Root = 0.975')
# Starts at (4.9, 0.07) to (1.02, 0.99)

plt.legend(loc="upper left")
plt.ylim(-1.5, 2.0)
plt.show()

plt.show()


#%%% Constructing the expectation of U for specific R0 value 

# dispersions = np.arange(0.1,1, 0.1)


# r0_125 = 1.25
# array_125 = np.empty((len(dispersions)))

# count = 0
# for d1 in dispersions:
#     array_125[count] = solvingForExpectation([r0_125, d1], sigma)
#     count += 1


# r0_2 = 2
# array_2 = np.empty((len(dispersions)))

# count = 0
# for d2 in dispersions:
#     array_2[count] = solvingForExpectation([r0_2, d2], sigma)
#     count += 1

# r0_275 = 4.5
# array_275 = np.empty((len(dispersions)))

# count = 0
# for d3 in dispersions:
#     array_275[count] = solvingForExpectation([r0_275, d3], sigma)
#     count += 1
    
# plt.scatter(dispersions, array_125) 

# plt.scatter(dispersions, array_2)


# plt.scatter(dispersions, array_275)

# plt.show()



