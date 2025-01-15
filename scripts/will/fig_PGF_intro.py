import numpy as np
import matplotlib.pyplot as plt
from stochastic_pgfs.random_graphs import poisson_degree_sequence
from stochastic_pgfs.laub_xia_algo import G1, get_outbreak_size

GREEN="#519E3E"
ORANGE="#EF8636"
TEAL="#37B1AB"
SALMON="#F77B61"
BLUE="#3B75AF"

# Style parameters
ORIGINAL_COLOR = '#2E5077'
SPINE_COLOR = '#333333'
GRID_ALPHA = 0.3
DIAGONAL_ALPHA = 0.5
LINEWIDTH = 3
FONT_SIZE_LABELS = 18
FONT_SIZE_TITLE = 20
TITLE_PAD = 20

def plot_stepped_distribution(x, y, ax=None, color='dodgerblue', linewidth=1):
    """Create a stepped distribution plot."""
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
    
    # Plot the line
    ax.plot(xs, ys, color=color, linewidth=linewidth)
    return ax

# Create figure with three panels
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8), facecolor='white')

# Generate Poisson degree sequence
lmbd = 1.5  # Mean degree
N = 10      # Maximum degree
my_poly_coef = poisson_degree_sequence(lmbd, N)
my_poly_coef = np.vstack((np.arange(0, my_poly_coef.shape[0], 1), my_poly_coef)).T.copy()

# Panel 1: Degree distribution
# plot_stepped_distribution(
#     my_poly_coef[:, 0],
#     my_poly_coef[:, 1],
#     ax=ax1,
#     color=ORANGE,
#     linewidth=LINEWIDTH
# )

ax1.bar(my_poly_coef[:, 0], my_poly_coef[:, 1], color=ORANGE, alpha=1,label='Degree Distribution')



# Panel 2: G1 function
x_min = 0
x_space = np.linspace(x_min, 1, 100)
y_vals = [G1(x_i, my_poly_coef, T=1) for x_i in x_space]

# Plot G1 curve
ax2.plot(x_space, y_vals, color=BLUE, linewidth=LINEWIDTH+2,label = r'$G_1(x)$')

# Find and plot fixed point
u1 = 0.5  # Initial guess
for _ in range(100):
    u2 = G1(u1, my_poly_coef, T=1)
    if abs(u2 - u1) < 1e-6:
        break
    u1 = u2
ax2.plot(u2, u2, 'o', color=TEAL, markersize=25, zorder=5,label = r'$u$')

# Add diagonal line y=x
ax2.plot([x_min, 1], [x_min, 1], 'k--', alpha=DIAGONAL_ALPHA,linewidth =LINEWIDTH,label = r'$y=x$')

# Panel 3: Outbreak size vs T
T_space = np.linspace(0, 1, 100)
outbreak_sizes = [get_outbreak_size(my_poly_coef[:, 1], T) for T in T_space]

# Add phase transition line
T_c = 1/lmbd  # Critical transmission probability
ax3.axvline(x=T_c, color='grey', linestyle='--', linewidth=LINEWIDTH, 
            label=r'Critical Transition at $T_c$', alpha=0.8)




ax3.plot(T_space, outbreak_sizes, color=GREEN, linewidth=LINEWIDTH+4, label='Outbreak Size')




# Customize all plots
for ax in (ax1, ax2, ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(SPINE_COLOR)
    ax.spines['bottom'].set_color(SPINE_COLOR)
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

# Add labels
ax1.set_xlabel(r'Degree($k$)', fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
ax1.set_ylabel(r'Probability($p_k$)', fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
ax1.set_title('Degree Distribution', fontsize=FONT_SIZE_TITLE, color=SPINE_COLOR, pad=TITLE_PAD)

ax2.set_xlabel(r'$x$', fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
ax2.set_ylabel(r'$G1(x)$', fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
ax2.set_title('Self-Consistent Equation', fontsize=FONT_SIZE_TITLE, color=SPINE_COLOR, pad=TITLE_PAD)
ax2.legend(fontsize=FONT_SIZE_LABELS + 2)

ax3.set_xlabel(r'Transmission Probability ($T$)', fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
ax3.set_ylabel('Outbreak Size', fontsize=FONT_SIZE_LABELS, color=SPINE_COLOR)
ax3.set_title('Outbreak Size', fontsize=FONT_SIZE_TITLE, color=SPINE_COLOR, pad=TITLE_PAD)
ax3.legend(fontsize=FONT_SIZE_LABELS + 2)



plt.tight_layout()
plt.savefig('figures/poster_fig1.svg')
