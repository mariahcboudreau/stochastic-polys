import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib as mpl 
import numpy as np

# Load your data
df = pd.read_csv("data/random_graphs/random_graphs_sweep_poisson_additive.csv")
df_pl = pd.read_csv("data/random_graphs/random_graphs_sweep_powerlaw_additive.csv")

# # Clean and filter data
df_pl = df_pl.query('lmbd > 2.0')
# df = df.query('lmbd > 1.0')  # Add minimum lambda filter for Poisson

# Filter non-critical values
def filter_non_critical_values(df):
    non_critical_t_values = df.groupby('T').agg(row_count=('lmbd', 'count')).reset_index()
    non_critical_t_values = non_critical_t_values.query('row_count > 1')['T']
    return df.query('T in @non_critical_t_values')

# Clean the data
df = filter_non_critical_values(df)
df_pl = filter_non_critical_values(df_pl)

# Cap extreme values
# df.loc[df['sce'] > 1, 'sce'] = 1
# df_pl.loc[df_pl['sce'] > 0.3, 'sce'] = 0.3

# Remove any remaining inf/nan values
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df_pl = df_pl.replace([np.inf, -np.inf], np.nan).dropna()

# Set plot parameters
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"

# Create the figure and subplots
fig, ax = plt.subplots(ncols=2, figsize=(15, 7), sharey=False)

# Create pivot tables with cleaned data
heatmap_data = df.pivot(index="lmbd", columns="T", values="sce").iloc[::-1]
contour_data = df.pivot(index="lmbd", columns="T", values="outbreak_size").iloc[::-1]
heatmap_data_pl = df_pl.pivot(index="lmbd", columns="T", values="sce").iloc[::-1]
contour_data_pl = df_pl.pivot(index="lmbd", columns="T", values="outbreak_size").iloc[::-1]

# Fill any remaining NaN values with interpolation
heatmap_data = heatmap_data.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
contour_data = contour_data.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
heatmap_data_pl = heatmap_data_pl.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
contour_data_pl = contour_data_pl.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)

# Calculate global limits
global_vmin = min(heatmap_data.min().min(), contour_data.min().min())
global_vmax = max(heatmap_data.max().max(), contour_data.max().max())

# Plot first heatmap
im = ax[0].imshow(heatmap_data, cmap="Greens", aspect='auto',
                  extent=[heatmap_data.columns.min(), heatmap_data.columns.max(),
                         heatmap_data.index.min(), heatmap_data.index.max()])

# Create meshgrid and plot contours for first plot
T_vals = heatmap_data.columns.values.astype(float)
lmbd_vals = heatmap_data.index.values.astype(float)
T_grid, lmbd_grid = np.meshgrid(T_vals, lmbd_vals)
CS = ax[0].contour(T_grid, lmbd_grid, contour_data.values, cmap='copper', linewidths=2)
ax[0].clabel(CS, inline=True, fontsize=8)
ax[0].set_title('Poisson')
ax[0].set_xlabel('Transmission Probability (T)')
ax[0].set_ylabel('Mean Degree (λ)')

# Add colorbar
cbar = fig.colorbar(im, ax=ax[0])
cbar.set_label('SCE')

# Plot second heatmap
im_pl = ax[1].imshow(heatmap_data_pl, cmap="Greens", aspect='auto',
                     extent=[heatmap_data_pl.columns.min(), heatmap_data_pl.columns.max(),
                            heatmap_data_pl.index.min(), heatmap_data_pl.index.max()])

# Create meshgrid and plot contours for second plot
T_vals_pl = heatmap_data_pl.columns.values.astype(float)
lmbd_vals_pl = heatmap_data_pl.index.values.astype(float)
T_grid_pl, lmbd_grid_pl = np.meshgrid(T_vals_pl, lmbd_vals_pl)
CS_pl = ax[1].contour(T_grid_pl, lmbd_grid_pl, contour_data_pl.values, cmap='copper', linewidth=2)
ax[1].clabel(CS_pl, inline=True, fontsize=8)
ax[1].set_title('Power Law')
ax[1].set_xlabel('Transmission Probability (T)')
ax[1].set_ylabel(r'Power Law Exponent($\alpha$)')

# Add colorbar
cbar_pl = fig.colorbar(im_pl, ax=ax[1])
cbar_pl.set_label('SCE')

plt.tight_layout()
plt.show()
#plt.savefig("figures/random_graphs_sweep_poisson_powerlaw_addative_heatmap.pdf")