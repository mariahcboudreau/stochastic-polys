#######
#
#   Implementing Laub and Xia algorithm
#
#######

##################### IMPORTS #########################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import schur, eigvals, companion
import math




##################### ESTABLISH COMPANION MATRIX #########################################

# Parameters
maxk = 100
r0 = 2
k = 0.8
a= 1/k


# Establish negative binomial dist
g1True = np.zeros(maxk)
for i in range(maxk):          
    g1True[i] = (math.gamma(i+k)/(math.factorial(i)*math.gamma(k)))*((a*r0)/(1+a*r0))**(i) * (1/(1 + a*r0))**(k)

companion = np.identity(maxk-1)
companion = np.insert(companion, 0, 0, axis = 0)  
companion = np.insert(companion, maxk-1, -g1True, axis = 1)


g1True = np.array([2,3,1])
companion = np.identity(len(g1True)-1)
companion = np.insert(companion, 0, 0, axis = 0)  
companion = np.insert(companion, len(g1True)-1, -g1True, axis = 1)



##################### SCHUR DECOMPOSITION #########################################

T, U = schur(companion)
print(T)