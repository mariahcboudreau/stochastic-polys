from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from numba import jit, config
import logging
from stochastic_pgfs.laub_xia_algo import (
    l_x_algo,
    G1_prime,
    get_outbreak_size,
    generate_sphere_point,
    _perturb_polynomial,
    G1
)
from stochastic_pgfs.random_graphs import (
    poisson_degree_sequence, 
    powerlaw_degree_sequence
)

from stochastic_pgfs.random_graphs import (
    poisson_degree_sequence, 
    powerlaw_degree_sequence
)

import networkx as nx

GREEN="#519E3E"
ORANGE="#EF8636"
TEAL="#37B1AB"
SALMON="#F77B61"
BLUE="#3B75AF"
YELLOW="#F8A602"

# Style parameters
ORIGINAL_COLOR = '#2E5077'
PERTURBED_COLOR = '#A8A8A8'
SPINE_COLOR = '#333333'
ORIGINAL_ALPHA = 1.0
PERTURBED_ALPHA = 0.3
GRID_ALPHA = 0.3
DIAGONAL_ALPHA = 0.5
ORIGINAL_LINEWIDTH = 5
PERTURBED_LINEWIDTH = 1.5
FONT_SIZE_LABELS = 12
FONT_SIZE_TITLE = 14
TITLE_PAD = 20

def plot_stepped_distribution(x, y, ax=None, color='dodgerblue', fill=False, fill_color='gold', alpha=1.0, linewidth=1):
    """
    Create a stepped distribution plot with optional fill.
    
    Parameters:
    -----------
    x : array-like
        x-coordinates
    y : array-like
        y-coordinates (heights)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes
    color : str, optional
        Color of the line
    fill : bool, optional
        Whether to fill below the curve
    fill_color : str, optional
        Color of the fill if fill=True
    alpha : float, optional
        Transparency of the fill
    linewidth : float, optional
        Width of the line
        
    Returns:
    --------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()
        
    # Create stepped coordinates
    xs = [x[0] - 0.5]  # Start with left edge
    ys = [0]           # Start at zero
    
    # Create intermediate points
    for i in range(len(x)):
        xs.extend([x[i] - 0.5, x[i] + 0.5])
        ys.extend([y[i], y[i]])
        
    # Add final point
    xs.append(x[-1] + 0.5)
    ys.append(0)
    
    # Plot the line with linewidth parameter
    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)
    
    # Optional fill
    if fill:
        ax.fill_between(xs, 0, ys, color=fill_color, alpha=alpha)
        
    return ax

lmbd = 1.5
delta = 0.5
K = 10
N = 10

my_poly_coef = poisson_degree_sequence(lmbd,N)
my_poly_coef = np.vstack((np.arange(0, my_poly_coef.shape[0], 1), my_poly_coef)).T.copy()

Z = np.random.randn(N, K)
Q, R = np.linalg.qr(Z, mode='reduced')
Z = np.ascontiguousarray(Q[:, :K])  # Ensure we only take K columns
K = Z.shape[1]  # Update K to match actual number of vectors

i = 1
perturbed_poly_coef_list = []
for i in range(K):
    alpha_i = Z[:, i]
    perturbed_poly_coef_list.append(_perturb_polynomial(my_poly_coef,delta,alpha_i,perturbation_type='additive'))
   
x_space = np.linspace(0,1,100) 

[G1(x_i, my_poly_coef,T = 1) for x_i in x_space]
plt.plot(x_space, [G1(x_i, my_poly_coef,T = 1) for x_i in x_space], label='Original')
    
    
    
    
fig = plt.figure(figsize=(32, 8), facecolor='white')
gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])
axes = [ax1, ax2, ax3, ax4]

# Panel 1: Distribution plots
# Plot perturbed polynomials
for perturbed_poly_coef_i in perturbed_poly_coef_list:
    plot_stepped_distribution(
        perturbed_poly_coef_i[:, 0],
        perturbed_poly_coef_i[:, 1],
        ax=axes[0],
        color=ORANGE,
        alpha=PERTURBED_ALPHA,
        linewidth=PERTURBED_LINEWIDTH
    )

# Plot original polynomial
plot_stepped_distribution(
    my_poly_coef[:, 0],
    my_poly_coef[:, 1],
    ax=axes[0],
    color=ORANGE,
    linewidth=ORIGINAL_LINEWIDTH
)

# Panel 2: G1 function plots

x_min = 0.1
x_space = np.linspace(x_min, 1, 100)

# Plot G1 for perturbed polynomials and find their fixed points
for perturbed_poly_coef_i in perturbed_poly_coef_list:
    # Plot G1 curve
    y_vals = [G1(x_i, perturbed_poly_coef_i, T=1) for x_i in x_space]
    axes[1].plot(x_space, y_vals, color=BLUE, alpha=PERTURBED_ALPHA,label = r'Pertrubed $G_1(x)$')
    
    # Find fixed point through iteration
    u1 = 0.5  # Initial guess
    for _ in range(100):  # Maximum iterations
        u2 = G1(u1, perturbed_poly_coef_i, T=1)
        if abs(u2 - u1) < 1e-6:
            break
        u1 = u2
    # Plot fixed point
    axes[1].plot(u2, u2, 'o', color=TEAL, alpha=PERTURBED_ALPHA, markersize=10)

# Plot G1 for original polynomial and its fixed point
y_vals_orig = [G1(x_i, my_poly_coef, T=1) for x_i in x_space]
axes[1].plot(x_space, y_vals_orig, color=BLUE, linewidth=ORIGINAL_LINEWIDTH)  # Increased linewidth

# Find fixed point for original polynomial
u1 = 0.5  # Initial guess
for _ in range(100):
    u2 = G1(u1, my_poly_coef, T=1)
    if abs(u2 - u1) < 1e-6:
        break
    u1 = u2
# Plot original fixed point with larger marker
axes[1].plot(u2, u2, 'o', color=TEAL, markersize=12, zorder=5)

# Add diagonal line y=x
axes[1].plot([x_min, 1], [x_min, 1], 'k--', alpha=DIAGONAL_ALPHA)

# Panel 3: Outbreak size vs T with confidence intervals
T_space = np.linspace(0, 1, 50)

# Calculate outbreak sizes for all perturbed polynomials
all_perturbed_outbreaks = np.zeros((len(perturbed_poly_coef_list), len(T_space)))
for i, perturbed_poly_coef_i in enumerate(perturbed_poly_coef_list):
    all_perturbed_outbreaks[i] = [get_outbreak_size(perturbed_poly_coef_i[:, 1], T) for T in T_space]

# Calculate percentiles for each T value
lower_bound = np.percentile(all_perturbed_outbreaks, 5, axis=0)
upper_bound = np.percentile(all_perturbed_outbreaks, 95, axis=0)

# Calculate outbreak sizes for original polynomial
outbreak_sizes_orig = [get_outbreak_size(my_poly_coef[:, 1], T) for T in T_space]

# Plot confidence interval
axes[2].fill_between(T_space, lower_bound, upper_bound, color=YELLOW, alpha=0.2, label='SCE')

# Plot original outbreak size
axes[2].plot(T_space, outbreak_sizes_orig, color=GREEN, linewidth=ORIGINAL_LINEWIDTH, label='Original')

# Add legend
axes[2].legend(fontsize=FONT_SIZE_LABELS)

# Customize third panel
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].spines['left'].set_color(SPINE_COLOR)
axes[2].spines['bottom'].set_color(SPINE_COLOR)
axes[2].grid(True, linestyle='--', alpha=GRID_ALPHA)
axes[2].set_axisbelow(True)

# Add labels for third panel
axes[2].set_xlabel('Transmission Probability (T)', fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
axes[2].set_ylabel('Outbreak Size', fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
axes[2].set_title('Outbreak Size vs. Transmission Probability', fontsize=FONT_SIZE_TITLE, color=SPINE_COLOR, pad=TITLE_PAD)

# Create branching process visualization
# Create a branching process with varying number of children
def create_branching_process(max_levels, max_children):
    G = nx.DiGraph()
    G.add_node(0)  # Root node
    current_level = [0]
    next_node = 1
    
    for level in range(max_levels):
        next_level = []
        for node in current_level:
            num_children = np.random.randint(0, max_children + 1)  # Allow for 0 children
            for _ in range(num_children):
                G.add_node(next_node)
                G.add_edge(node, next_node)
                next_level.append(next_node)
                next_node += 1
        if not next_level:  # If no nodes in next level, stop
            break
        current_level = next_level
    return G

# Create the branching process graph
G = create_branching_process(max_levels=3, max_children=2)

# Create hierarchical layout
def hierarchical_layout(G, root=0):
    pos = {}
    # Get all nodes at each level
    levels = {level: [] for level in range(max_levels)}
    for node in G.nodes():
        try:
            level = nx.shortest_path_length(G, root, node)
            levels[level].append(node)
        except nx.NetworkXNoPath:
            continue
    
    # Position nodes level by level
    for level, nodes in levels.items():
        if not nodes:
            continue
        # Calculate horizontal spacing
        width = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = (i - (width - 1)/2) * 0.5  # Horizontal position
            y = 1 - level * 0.3            # Vertical position
            pos[node] = np.array([x, y])
    
    return pos

# Get positions
pos = hierarchical_layout(G)

# Create a new figure for the branching process visualization
fig_bp = plt.figure(figsize=(8, 8), facecolor='white')
ax_bp = fig_bp.add_subplot(111)

# Draw the network
nx.draw_networkx_edges(G, pos, ax=ax_bp, 
                      edge_color=SALMON, 
                      width=3, 
                      alpha=0.8,
                      arrowsize=20,
                      arrowstyle='->')

nx.draw_networkx_nodes(G, pos, ax=ax_bp,
                      node_color=TEAL,
                      node_size=1500,
                      alpha=1)

# Remove axis and set title
ax_bp.set_xticks([])
ax_bp.set_yticks([])
ax_bp.set_title('Branching Process', fontsize=FONT_SIZE_TITLE, color=SPINE_COLOR, pad=TITLE_PAD)
for spine in ax_bp.spines.values():
    spine.set_visible(False)

# Save the branching process figure
fig_bp.savefig("figures/branching_process_viz.svg", bbox_inches='tight')

# Customize all plots
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(SPINE_COLOR)
    ax.spines['bottom'].set_color(SPINE_COLOR)
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

# Add labels
for i, (xlabel, ylabel, title) in enumerate([
    ('Degree', 'Probability', 'Polynomial Coefficient Perturbations'),
    ('x', 'G1(x)', 'G1 Function'),
    ('Transmission Probability (T)', 'Outbreak Size', 'Outbreak Size vs. Transmission Probability')
]):
    axes[i].set_xlabel(xlabel, fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
    axes[i].set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
    axes[i].set_title(title, fontsize=FONT_SIZE_TITLE, color=SPINE_COLOR, pad=TITLE_PAD)

plt.tight_layout()

plt.savefig("figures/fig1_viz.svg")
fig_bp.savefig("figures/branching_process_viz.svg")