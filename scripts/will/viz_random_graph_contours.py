import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib as mpl 
import numpy as np

# Load your data
df = pd.read_csv("data/random_graphs/random_graphs_sweep_poisson_addative.csv")
df_pl = pd.read_csv("data/random_graphs/random_graphs_sweep_powerlaw_addative.csv")

# Filter non-critical values
def filter_non_critical_values(df):
    non_critical_t_values = df.groupby('T').agg(row_count=('lmbd', 'count')).reset_index().query('row_count > 1')['T']
    return df.query('T in @non_critical_t_values')

df = filter_non_critical_values(df)
df = df.query('T > 0.4 & lmbd > 1.0')



df_pl = filter_non_critical_values(df_pl)

# Set plot parameters
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"

# Create the figure and subplots
fig, ax = plt.subplots(ncols=2, figsize=(15, 7), sharey=False)

# Heatmap and contour plot for df
heatmap_data = df.pivot(index="lmbd", columns="T", values="sce").iloc[::-1]  # flip the rows
contour_data = df.pivot(index="lmbd", columns="T", values="outbreak_size").iloc[::-1]  # flip the rows

# Calculate the global vmin and vmax for both heatmap and contour data
global_vmin = min(heatmap_data.min().min(), contour_data.min().min())
global_vmax = max(heatmap_data.max().max(), contour_data.max().max())

# Plot the heatmap using imshow
im = ax[0].imshow(heatmap_data, cmap="Greens", aspect='auto', vmin=global_vmin, vmax=global_vmax,
                  extent=[heatmap_data.columns.min(), heatmap_data.columns.max(), heatmap_data.index.min(), heatmap_data.index.max()])

# Create meshgrid with correct axes
T_vals = heatmap_data.columns.values.astype(float)
lmbd_vals = heatmap_data.index.values.astype(float)
T_grid, lmbd_grid = np.meshgrid(T_vals, lmbd_vals)

# Plot the contour
CS = ax[0].contour(T_grid, lmbd_grid, contour_data.values, cmap='copper', vmin=heatmap_data.index.min(), vmax=heatmap_data.index.max())
ax[0].clabel(CS, inline=True, fontsize=8)
ax[0].set_title('Poisson')
ax[0].set_xlabel('Transmission Probability (T)')
ax[0].set_ylabel('Mean Degree (λ)')

# Add colorbar
cbar = fig.colorbar(im, ax=ax[0])
cbar.set_label('SCE')

# Repeat the corrections for df_pl
heatmap_data_pl = df_pl.pivot(index="lmbd", columns="T", values="sce").iloc[::-1]  # flip the rows
contour_data_pl = df_pl.pivot(index="lmbd", columns="T", values="outbreak_size").iloc[::-1]  # flip the rows

# Plot the heatmap using imshow
im_pl = ax[1].imshow(heatmap_data_pl, cmap="Greens", aspect='auto', vmin=global_vmin, vmax=global_vmax,
                     extent=[heatmap_data_pl.columns.min(), heatmap_data_pl.columns.max(), heatmap_data_pl.index.min(), heatmap_data_pl.index.max()])

# Create meshgrid with correct axes
T_vals_pl = heatmap_data_pl.columns.values.astype(float)
lmbd_vals_pl = heatmap_data_pl.index.values.astype(float)
T_grid_pl, lmbd_grid_pl = np.meshgrid(T_vals_pl, lmbd_vals_pl)

# Plot the contour
CS_pl = ax[1].contour(T_grid_pl, lmbd_grid_pl, contour_data_pl.values, cmap='copper', vmin=global_vmin, vmax=global_vmax)
ax[1].clabel(CS_pl, inline=True, fontsize=8)
ax[1].set_title('Power Law')
ax[1].set_xlabel('Transmission Probability (T)')
ax[1].set_ylabel(r'Power Law Exponent($\alpha$)')

# Add colorbar
cbar_pl = fig.colorbar(im_pl, ax=ax[1])
cbar_pl.set_label('SCE')

plt.tight_layout()
plt.savefig("figures/random_graphs_sweep_poisson_powerlaw_addative_heatmap.pdf")