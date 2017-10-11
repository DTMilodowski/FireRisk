#==============================================================================
# plot_FWI.py
#
# David T. Milodowski
# October 2017
#
# This is a set of plotting functions to facilitate analysis of the FWI metrics
#============================================================================== 
import numpy as np
from matplotlib import pyplot as plt
import sys

# Get perceptually uniform colourmaps
sys.path.append('/home/dmilodow/DataStore_DTM/FOREST2020/EOdata/EO_data_processing/src/plot_EO_data/colormap/')
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.set_cmap(cmaps.inferno) # set inferno as default - a good fiery one :-)
