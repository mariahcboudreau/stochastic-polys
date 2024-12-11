import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib as mpl 
from matplotlib.gridspec import GridSpec
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from stochastic_pgfs.laub_xia_algo import (
        G1_prime
)

from stochastic_pgfs.random_graphs import (
    er_degree_sequence,
    powerlaw_degree_sequence,
    poisson_degree_sequence,
    make_config_model,
    plot_graph,
    mean_power_law
)


# Each process reads the data
df = pd.read_csv("data/random_graphs/random_graphs_sweep_poisson_additive.csv")
df_pl = pd.read_csv("data/random_graphs/random_graphs_sweep_powerlaw_additive.csv")
df_pl = df_pl.query('lmbd > 2.0')

N_max = 100

def calculate_critical_transition(my_degree_sequence):
    pk = np.vstack((np.arange(0, my_degree_sequence.shape[0], 1), my_degree_sequence)).T
    return 1/G1_prime(1, pk, 1)

def power_law_critical_transition(alpha):
        power_law_degree_seq = powerlaw_degree_sequence(alpha = alpha,N_max = N_max)
        return calculate_critical_transition(power_law_degree_seq)

def poisson_critical_transition(lmbd):
        poisson_degree_seq = poisson_degree_sequence(lmbd, N_max = N_max)
        return calculate_critical_transition(poisson_degree_seq)


# Calculate critical transitions in parallel
powerlaw_params = df_pl['lmbd'].unique()
poisson_params = df['lmbd'].unique()

powerlaw_critical_transition = pd.Series([power_law_critical_transition(gamma) for gamma in powerlaw_params])
poisson_critical_transition = pd.Series([poisson_critical_transition(gamma) for gamma in poisson_params])

powerlaw_critical_transition = pd.Series(np.where(powerlaw_critical_transition > 1, 1, powerlaw_critical_transition))
poisson_critical_transition = pd.Series(np.where(poisson_critical_transition > 1, 1, poisson_critical_transition))

# Only root process creates the visualization
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"

# Choose a colormap
cmap = plt.get_cmap('copper')  # You can replace 'viridis' with any other continuous colormap
# Extract the first color
first_color = cmap(10)  # 0 corresponds to the start of the colormap
first_color = 'blue'
lw =0.7 

#graph config options 
edge_options = {
        'edge_color': 'black',
        'width' :0.5 
}

node_options = {
        'node_size' :30 ,
        'edgecolors' : 'black',
        'node_color': 'white',
        'linewidths' : 0.5
}

gs = GridSpec(3, 2, wspace=0.2, hspace=0.2,height_ratios = [1,1,1])
fig = plt.figure(figsize=(15, 15))
plt.subplots_adjust(left=0.12, right=0.84, bottom=0.1, top=0.95, wspace=0.4, hspace=0.4)

ax_g1 = fig.add_subplot(gs[0,0])
ax_g1.set_title('$Erd\H{o}s-Rényi$',fontsize = 24)

p = 0.5
n = 80
num_nodes =200

G = make_config_model(er_degree_sequence,p = p,n = n,num_nodes = num_nodes)
plot_graph(G,node_options,edge_options,ax_g1)

ax_g2 = fig.add_subplot(gs[0,1])
ax_g2.set_title('Power-law CM',fontsize = 24)

alpha = 1.1
N_max = 100

G = make_config_model(powerlaw_degree_sequence,alpha = alpha,N_max = N_max,num_nodes = num_nodes)
plot_graph(G,node_options,edge_options,ax_g2)

ax1 = fig.add_subplot(gs[1,0])

sns.lineplot(data = df,x = 'T',y = 'outbreak_size',hue = 'lmbd',ax = ax1,color = first_color,palette = 'copper')
# Add vertical lines for Poisson critical transitions
for lmbd, crit in zip(df['lmbd'].unique(), poisson_critical_transition.unique()):
        ax1.axvline(x=crit, color='gray', linestyle=':', alpha=0.5)

ax1.set(ylabel = 'Outbreak Size',
        xlabel = 'Transmission Probability(T)',
        title = 'Poisson Outbreak Size vs. Transmission Probability'
        )
ax1.legend_.remove()

ax2 = fig.add_subplot(gs[2,0])

sns.lineplot(data = df,x = 'T',y = 'sce',hue = 'lmbd',ax = ax2,color = first_color,palette = 'copper',lw = lw)
# Add vertical lines for Poisson critical transitions
for lmbd, crit in zip(df['lmbd'].unique(), poisson_critical_transition.unique()):
        ax2.axvline(x=crit, color='gray', linestyle=':', alpha=0.5)

ax2.set(ylabel = 'Condition Number',xlabel = 'Transmission Probability(T)',title = 'Poisson Condition Number vs. Transmission Probability(T)')

ax2.legend_.remove()

ax3 = fig.add_subplot(gs[1,1])
sns.lineplot(data = df_pl,x = 'T',y = 'outbreak_size',hue = 'lmbd',ax = ax3,color = first_color,palette = 'copper')
# Add vertical lines for Power Law critical transitions
for alpha, crit in zip(df_pl['lmbd'].unique(), powerlaw_critical_transition.unique()):
        ax3.axvline(x=crit, color='gray', linestyle=':', alpha=0.5)

ax3.set(ylabel = 'Outbreak Size',
        xlabel = 'Transmission Probability(T)',
        title = 'Power Law Outbreak Size vs. Transmission Probability'
        )

ax3.legend_.remove()
#ax3.legend(title = r'$\alpha$')

ax4 = fig.add_subplot(gs[2,1])
sns.lineplot(data = df_pl,x = 'T',y = 'sce',hue = 'lmbd',ax = ax4,color = first_color,palette = 'copper',lw = lw)
# Add vertical lines for Power Law critical transitions
for alpha, crit in zip(df_pl['lmbd'].unique(), powerlaw_critical_transition.unique()):
        ax4.axvline(x=crit, color='gray', linestyle=':', alpha=0.5)

ax4.set(ylabel = 'Condition Number',xlabel = 'Transmission Probability(T)',title = ' Power Law Condition Number vs. Transmission Probability(T)')
#ax4.legend_.remove()
#ax4.legend(title = r'$\alpha$')

#add a colorbar for ER
ax4.axvline(x=crit, color='gray', linestyle=':', alpha=0.5)

ax4.set(ylabel = 'Condition Number',xlabel = 'Transmission Probability(T)',title = ' Power Law Condition Number vs. Transmission Probability(T)')
ax4.legend_.remove()
#ax4.legend(title = r'$\alpha$')

#add a colorbar for ER
norm = mpl.colors.Normalize(vmin = df['lmbd'].min(),vmax = df['lmbd'].max())
sm_er = plt.cm.ScalarMappable(cmap = 'copper',norm =norm)
sm_er.set_array([])
#add colorbar for PL
norm = mpl.colors.Normalize(vmin = df_pl['lmbd'].min(),vmax = df_pl['lmbd'].max())
sm_pl = plt.cm.ScalarMappable(cmap = 'copper',norm =norm)
sm_pl.set_array([])


cbar_frac = 0.09
# Add the colorbar to the figure
cbar1 = fig.colorbar(sm_er, ax=ax1, orientation='vertical', fraction=cbar_frac, pad=0.04)
cbar2 = fig.colorbar(sm_er, ax=ax2, orientation='vertical', fraction=cbar_frac, pad=0.04)

cbar3 = fig.colorbar(sm_pl, ax=ax3, orientation='vertical', fraction=cbar_frac, pad=0.04)
cbar4 = fig.colorbar(sm_pl, ax=ax4, orientation='vertical', fraction=cbar_frac, pad=0.04)

cbar1.set_label(r'$\lambda$')
cbar2.set_label(r'$\lambda$')

cbar3.set_label(r'$\alpha$')
cbar4.set_label(r'$\alpha$')

plt.show()
#plt.savefig("figures/condition_number_vs_mean_degree_er_powerlaw_addative.png")
#plt.savefig("figures/condition_number_vs_mean_degree_er_powerlaw_addative.pdf")

