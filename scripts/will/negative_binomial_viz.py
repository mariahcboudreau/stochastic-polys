import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom

# Colors from your scheme
TEAL = "#37B1AB"
ORANGE="#EF8636"
SPINE_COLOR = '#333333'

# Style parameters
FONT_SIZE_LABELS = 14
FONT_SIZE_TITLE = 16
TITLE_PAD = 20

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

# Parameters for negative binomial
n = 5  # number of successes
p = 0.5  # probability of success
max_k = 20  # maximum number of trials to plot

# Generate x values and probabilities
k = np.arange(0, max_k)
prob = nbinom.pmf(k, n, p)

# Create the bar plot
ax.bar(k, prob, color=ORANGE, alpha=0.8, width=0.8)

# Remove all spines, ticks, and labels
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Number of Offspring', fontsize=FONT_SIZE_LABELS)
ax.set_ylabel('Probability')
ax.set_title('Negative Binomial')

# Add grid
ax.grid(False)  # Remove grid as well
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
#plt.savefig("figures/negative_binomial.svg", bbox_inches='tight')
plt.close()
