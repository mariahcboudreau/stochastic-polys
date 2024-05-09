#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:06:05 2023

@author: mariahboudreau
"""


#######
## PROCESSING VACC CONTOUR OUTPUT
##
#######

import numpy as np  
import pandas as pd
from glob import glob

r0_thresh = 1.8
k_thresh = 0.1


#Reading files 
contour_dfs = []
for outputfile in glob("./csvs/heatmap_results_3-13-23_*.csv"): # df['r'] = 3 for the first run
    contour_df = pd.read_csv(outputfile, index_col = 0)
    contour_dfs.append(contour_df)
    
contour_df = pd.concat(contour_dfs)

contour_df = contour_df.sort_values(by = ['K','R0'], ascending=[True, True])

contour_df['Delta*True Root'] = contour_df['Expection Delta U']*contour_df['True Root']

contour_df['Division']= contour_df['Expection Delta U']/contour_df['Root']

contour_df['Division out']= contour_df['Expection Delta U']/contour_df['Outbreak']


df_roots = contour_df[['R0', 'K', 'Root']]
df_roots = df_roots[df_roots['R0'] >= r0_thresh]
df_roots = df_roots[df_roots['K'] >= k_thresh] 
df_roots = df_roots.reset_index() 
df_roots = df_roots.drop(['index'], axis = 1)          
df_roots_avg = df_roots.groupby(['R0', 'K']).mean().reset_index()
df_roots_std = df_roots.groupby(['R0', 'K']).std().reset_index()



df_true_roots = contour_df[['R0', 'K', 'True Root']]
df_true_roots = df_true_roots[df_true_roots['R0'] >= r0_thresh]
df_true_roots = df_true_roots[df_true_roots['K'] >= k_thresh]
df_true_roots = df_true_roots.reset_index()  
df_true_roots = df_true_roots.drop(['index'], axis = 1)
df_true_roots_avg = df_true_roots.groupby(['R0', 'K']).mean().reset_index()  



df_true_out = contour_df[['R0', 'K', 'True Outbreak']]
df_true_out = df_true_out[df_true_out['R0'] >= r0_thresh]
df_true_out = df_true_out[df_true_out['K'] >= k_thresh]
df_true_out = df_true_out.reset_index()  
df_true_out = df_true_out.drop(['index'], axis = 1)
df_true_out_avg = df_true_out.groupby(['R0', 'K']).mean().reset_index() 



df_outbreak = contour_df[['R0', 'K', 'Outbreak']] 
df_outbreak = df_outbreak[df_outbreak['R0'] >= r0_thresh]
df_outbreak = df_outbreak[df_outbreak['K'] >=  k_thresh]
df_outbreak = df_outbreak.reset_index() 
df_outbreak = df_outbreak.drop(['index'], axis = 1) 
df_outbreak_avg = df_outbreak.groupby(['R0', 'K']).mean().reset_index()   





df_delta_roots = contour_df[['R0', 'K', 'Expection Delta U']]
df_delta_roots = df_delta_roots[df_delta_roots['R0'] >= r0_thresh]
df_delta_roots = df_delta_roots[df_delta_roots['K'] >=  k_thresh]
df_delta_roots = df_delta_roots.reset_index()
df_delta_roots = df_delta_roots.drop(['index'], axis = 1)
df_delta_roots_avg = df_delta_roots.groupby(['R0','K']).mean().reset_index()


df_delta_scaled = contour_df[['R0', 'K', 'Delta*True Root']]
df_delta_scaled = df_delta_scaled[df_delta_scaled['R0'] >= r0_thresh]
df_delta_scaled = df_delta_scaled[df_delta_scaled['K'] >=  k_thresh]
df_delta_scaled = df_delta_scaled.reset_index()
df_delta_scaled = df_delta_scaled.drop(['index'], axis = 1)
df_delta_scaled_avg = df_delta_scaled.groupby(['R0','K']).mean().reset_index()



df_division_roots = contour_df[['R0', 'K', 'Division']] 
df_division_roots = df_division_roots[df_division_roots['R0'] >= r0_thresh]
df_division_roots = df_division_roots[df_division_roots['K'] >=  k_thresh]
df_division_roots = df_division_roots.reset_index()
df_division_roots = df_division_roots.drop(['index'], axis = 1)
df_division_roots_avg = df_division_roots.groupby(['R0', 'K']).mean().reset_index()



df_division_outbreak = contour_df[['R0', 'K', 'Division out']] 
df_division_outbreak['Division out log'] = np.log(df_division_outbreak['Division out'])
df_division_outbreak = df_division_outbreak[df_division_outbreak['R0'] >= r0_thresh]
df_division_outbreak = df_division_outbreak[df_division_outbreak['K'] >= k_thresh]
df_division_outbreak = df_division_outbreak.reset_index()
df_division_outbreak = df_division_outbreak.drop(['index'], axis = 1)
df_division_outbreak_avg = df_division_outbreak.groupby(['R0', 'K']).mean().reset_index()


df_division_outbreak_avg['STD/out'] = df_roots_std['Root']/df_outbreak_avg['Outbreak']  
df_division_outbreak_avg['STD/out log'] = np.log(df_division_outbreak_avg['STD/out']) 


### AVERAGE THE RUNS HERE BEFORE PIVOTING

pivot_root = df_roots_avg.pivot(index='K', columns = 'R0', values = 'Root')

pivot_root_std = df_roots_std.pivot(index='K', columns = 'R0', values = 'Root')

pivot_outbreak = df_outbreak_avg.pivot(index='K', columns = 'R0', values = 'Outbreak')\
    
pivot_true_root = df_true_roots_avg.pivot(index='K', columns = 'R0', values = 'True Root')

pivot_true_out = df_true_out_avg.pivot(index='K', columns = 'R0', values = 'True Outbreak')

pivot_division = df_division_roots_avg.pivot(index='K', columns = 'R0', values = 'Division')

pivot_division_out = df_division_outbreak_avg.pivot(index='K', columns = 'R0', values = 'Division out log')

pivot_division_out_std = df_division_outbreak_avg.pivot(index='K', columns = 'R0', values = 'STD/out log')

pivot_delta = df_delta_roots_avg.pivot(index='K', columns = 'R0', values = 'Expection Delta U')

pivot_delta_scaled = df_delta_scaled_avg.pivot(index='K', columns = 'R0', values = 'Delta*True Root')


#%% heatmap - analytical


import matplotlib as mpl
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

mpl.rcParams.update(mpl.rcParamsDefault)


fig, axs = plt.subplots(2,2, sharex=True, sharey= False)


####### SEPARATE COLORMAP #########
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(
        norm = norm,
        cmap=cm.viridis
    )
#fig.colorbar(sm, ax = axs[0,1], orientation='vertical')
#fig.colorbar(sm, ax = axs[1,1], orientation='vertical')


####### CONTOUR PLOT FOR ROOT #########


sns.heatmap(pivot_true_root, cmap=cm.Purples, vmin=0, vmax=1, ax = axs[0,0])

axs[0,0].set_title('True Root')
axs[0,0].set_ylabel('k, dispersion parameter')   
axs[0,0].set_xlabel('')

####### CONTOUR PLOT FOR OUTBREAK #########

sns.heatmap(pivot_true_out, cmap=cm.Purples, vmin=0, vmax=1, ax = axs[0,1])
axs[0,1].set_title('True outbreak size')
#axs[0,1].clabel(con_outbreak, inline=1, fontsize=6)
axs[0,1].set_xlabel('')
axs[0,1].set_ylabel('')


###### CONTOUR PLOT FOR DELTA ROOT #######



sns.heatmap(pivot_delta, cmap=cm.Oranges, ax = axs[1,0]) #### Do this without that strip
axs[1,0].set_title('E[$\Delta$ Root]')

# sns.heatmap(pivot_delta_scaled, cmap=cm.Oranges, ax = axs[1,0])
# axs[1,0].set_title('E[$\Delta$ Root]*True Root')

#axs[1,0].clabel(con_delta, inline=1, fontsize=6)

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

sns.heatmap(pivot_division_out, cmap=cm.PuOr, ax = axs[1,1]) ## CHANGE THIS TO NOT INCLUDE THE SMALL K VALUES



# #con_div_o = axs[1,1].contour(x_div_o, y_div_o, Z, cmap=cm.viridis, norm =norm, levels=levels)
# heat_div_o = axs[1,1].imshow(np.flipud(np.log(Z)), cm.viridis, aspect = 'auto')
axs[1,1].set_title('E[$\Delta$ Root]/True Outbreak')
#axs[1,1].clabel(con_div_o, inline=1, fontsize=6)

axs[1,1].set_xlabel('$R_{0}$, average secondary degree')
axs[1,1].set_ylabel('')

plt.show()
    
    



#plt.savefig("analytical_heats_3-22-23_vacc.pdf", format = 'pdf')


#%% heatmap - simulated



fig, axs = plt.subplots(2,2, sharex=True, sharey= False)


####### SEPARATE COLORMAP #########
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(
        norm = norm,
        cmap=cm.viridis
    )
#fig.colorbar(sm, ax = axs[0,1], orientation='vertical')
#fig.colorbar(sm, ax = axs[1,1], orientation='vertical')


####### CONTOUR PLOT FOR ROOT #########


sns.heatmap(pivot_root, cmap=cm.Reds, vmin=0, vmax=1, ax = axs[0,0])

axs[0,0].set_title('Root')
axs[0,0].set_ylabel('k, dispersion parameter')   
axs[0,0].set_xlabel('')

####### CONTOUR PLOT FOR OUTBREAK #########

sns.heatmap(pivot_outbreak, cmap=cm.Reds, vmin=0, vmax=1, ax = axs[0,1])
axs[0,1].set_title('Outbreak size')
#axs[0,1].clabel(con_outbreak, inline=1, fontsize=6)
axs[0,1].set_xlabel('')
axs[0,1].set_ylabel('')


###### CONTOUR PLOT FOR DELTA ROOT #######



sns.heatmap(pivot_root_std, cmap=cm.Blues, ax = axs[1,0]) #### Do this without that strip

axs[1,0].set_title('Standard Deviation of Root')
#axs[1,0].clabel(con_delta, inline=1, fontsize=6)

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

sns.heatmap(pivot_division_out_std, cmap=cm.RdBu, ax = axs[1,1]) ## CHANGE THIS TO NOT INCLUDE THE SMALL K VALUES



# #con_div_o = axs[1,1].contour(x_div_o, y_div_o, Z, cmap=cm.viridis, norm =norm, levels=levels)
# heat_div_o = axs[1,1].imshow(np.flipud(np.log(Z)), cm.viridis, aspect = 'auto')
axs[1,1].set_title('SD/Outbreak')
#axs[1,1].clabel(con_div_o, inline=1, fontsize=6)

axs[1,1].set_xlabel('$R_{0}$, average secondary degree')
axs[1,1].set_ylabel('')

plt.show()
    
    
# Questions for Laurent

# What values should we focus on for the parameters, is it okay to have the relative root on a different scale? 

# Inverse is different



#plt.savefig("simulated_heats_3-22-23_vacc.pdf", format = 'pdf')
