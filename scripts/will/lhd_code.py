import numpy as np
import matplotlib.pyplot as plt
from stochastic_pgfs.random_graphs import poisson_degree_sequence,powerlaw_degree_sequence
from stochastic_pgfs.laub_xia_algo import l_x_algo, iterate_until_convergence,G0,G1
from tqdm import tqdm
import time


lmbda = 2
T = np.float64(0.5)
#deg_seq = poisson_degree_sequence(lmbda)
#deg_seq = powerlaw_degree_sequence(2.0,100)


# l_x_algo(deg_seq,T= T)
pk = np.vstack((np.arange(0, deg_seq.shape[0], 1), deg_seq)).T
iterate_until_convergence(pk,T = T )


# def compute_outbreak(pk, Tlist, tol=1e-5):
#     usol = []
#     Pext = []
#     for T in reversed(Tlist):
#         u1, u2 = iterate_until_convergence(pk, T, tol)
#         usol.append(u2)
#         Pext.append(1 - G0(u2, pk, T))
#     return usol, Pext
    



Tlist = np.linspace(0.0,1.0,80)
sce_list = []
time_list = []

for T in tqdm(Tlist):
    start_time = time.time()
    sce_list.append(l_x_algo(deg_seq, T=T, K=10000))
    end_time = time.time()
    iteration_time = end_time - start_time
    time_list.append(iteration_time)
usol,Pext = compute_outbreak(pk,Tlist)
    
Pext = list(reversed(Pext))


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

ax1.plot(Tlist,Pext)
ax1.vlines(1/lmbda,0,1,ls='--',color='grey') #critical transmissibility
ax1.set(xlabel=r"Transmission Probability $T$",ylabel=r"Epidemic size",title="Epidemic size vs Transmission Probability")
ax2.set(xlabel=r"Transmission Probability $T$",ylabel=r"SCE",title="SCE")
# ax1.xlabel(r"Transmission Probability $T$")
# ax1.ylabel(r"Epidemic size")

ax2.plot(Tlist,sce_list)
plt.show()

plt.plot(Tlist,time_list)
plt.xlabel("Transmission Probability")
plt.ylabel("Runtime of Iterated Root Finder Time (s)")