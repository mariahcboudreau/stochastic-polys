# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:11:05 2021
"""

from matplotlib.colors import LinearSegmentedColormap

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pylab as pylab


GREEN="#519E3E"
ORANGE="#EF8636"
TEAL="#37B1AB"
SALMON="#F77B61"
BLUE="#3B75AF"
GRAY="#CCCCCC"

# Get the two endpoints of the cividis colormap as hex values
cividis = mpl.cm.get_cmap('cividis')
CIVIDIS_START = mpl.colors.rgb2hex(cividis(0))
CIVIDIS_END = mpl.colors.rgb2hex(cividis(230))
#crete a custom colormap from these
n_bins = 100

colors_blue = [(0, GRAY), (1, CIVIDIS_START)]  # Define start (0) and end (1) points
cmap_blue = LinearSegmentedColormap.from_list("custom_blue", colors_blue, N=n_bins)

# Create colormap from gray to salmon (changed from green)
colors_yellow = [(0, GRAY), (1, CIVIDIS_END)]  # Define start (0) and end (1) points
cmap_yellow = LinearSegmentedColormap.from_list("custom_yellow", colors_yellow, N=n_bins)


# color styling
def set_colors(n_colors=2):
    global cmap
    global pallette
    #cmap = "cmr.redshift"
    cmap = "cmr.emerald"
    qualitative_cmap = cmr.get_sub_cmap(cmap, 0.2, 0.9, N=n_colors)

    pallette = qualitative_cmap.colors
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=pallette)
    
 
    


def set_fonts(extra_params={}):
    params = {
        "font.family": "Sans-Serif",
        "font.sans-serif": ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"],
        "mathtext.fontset": "cm",
        "legend.fontsize": 12,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "figure.titlesize": 20,
    }
    for key, value in extra_params.items():
        params[key] = value
    pylab.rcParams.update(params)
