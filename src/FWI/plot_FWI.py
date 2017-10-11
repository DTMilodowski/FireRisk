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
from matplotlib import rcParams
import sys

# Set up some basiic parameters for the plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['arial']
rcParams['font.size'] = 8
rcParams['legend.numpoints'] = 1
axis_size = rcParams['font.size']+2

# Get perceptually uniform colourmaps
sys.path.append('/home/dmilodow/DataStore_DTM/FOREST2020/EOdata/EO_data_processing/src/plot_EO_data/colormap/')
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.set_cmap(cmaps.inferno) # set inferno as default - a good fiery one :-)

# Plot maps for all FWI indices at specified timestep
def plot_FWI_indices_for_tstep(FFMC,DMC,DC,ISI,BUI,FWI,tstep):

    plt.figure(1, facecolor='White',figsize=[8,12])

    # Plot a -> the FFMC
    ax1a = plt.subplot2grid((3,2),(0,0))
    ax1a.annotate('a - FFMC', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    ax1a.set_ylabel('Latitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot b -> the DMC
    ax1b = plt.subplot2grid((3,2),(1,0))
    ax1b.annotate('b - DMC', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    ax1b.set_ylabel('Latitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot c -> the DC
    ax1c = plt.subplot2grid((3,2),(2,0))
    ax1c.annotate('c - DC', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    ax1c.set_ylabel('Latitude',fontsize=axis_size)
    ax1c.set_xlabel('Longitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot d -> the ISI
    ax1d = plt.subplot2grid((3,2),(0,1))
    ax1d.annotate('d - ISI', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot e -> the BUI
    ax1e = plt.subplot2grid((3,2),(1,1))
    ax1e.annotate('e - BUI', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot f -> the FWI
    ax1f = plt.subplot2grid((3,2),(2,1))
    ax1f.annotate('f - FWI', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    ax1f.set_xlabel('Longitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    plt.show()
    return 0


# Plot time series for all FWI indices at specified timestep
def plot_FWI_indices_time_series_for_pixel(FFMC,DMC,DC,ISI,BUI,FWI,start_tstep,end_tstep,row,col):

    plt.figure(1, facecolor='White',figsize=[8,12])

    # Plot a -> the FFMC
    ax1a = plt.subplot2grid((6,1),(0,0))
    ax1a.annotate('a - FFMC', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    ax1a.set_ylabel('Latitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot b -> the DMC
    ax1b = plt.subplot2grid((6,1),(1,0))
    ax1b.annotate('b - DMC', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    ax1b.set_ylabel('Latitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot c -> the DC
    ax1c = plt.subplot2grid((6,1),(2,0))
    ax1c.annotate('c - DC', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    ax1c.set_ylabel('Latitude',fontsize=axis_size)
    ax1c.set_xlabel('Longitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot d -> the ISI
    ax1d = plt.subplot2grid((6,1),(3,0))
    ax1d.annotate('d - ISI', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot e -> the BUI
    ax1e = plt.subplot2grid((6,1),(4,0))
    ax1e.annotate('e - BUI', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    # Plot f -> the FWI
    ax1f = plt.subplot2grid((6,1),(5,0))
    ax1f.annotate('f - FWI', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)
    ax1f.set_xlabel('Longitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    plt.show()
    return 0
