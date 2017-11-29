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
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

import sys

# Set up some basiic parameters for the plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['arial']
rcParams['font.size'] = 8
rcParams['legend.numpoints'] = 1
axis_size = rcParams['font.size']+2
colour = ['#46E900','#1A2BCE','#E0007F']

# Get perceptually uniform colourmaps
sys.path.append('/home/dmilodow/DataStore_DTM/FOREST2020/EOdata/EO_data_processing/src/plot_EO_data/colormap/')
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.set_cmap(cmaps.inferno) # set inferno as default - a good fiery one :-)

# Plot maps for all FWI indices at specified timestep
def plot_FWI_indices_for_tstep(FFMC,DMC,DC,ISI,BUI,FWI,tstep):

    fig = plt.figure(1, facecolor='White',figsize=[10,8])

    # Plot a -> the FFMC
    ax1a = plt.subplot2grid((3,2),(0,0))
    ax1a.annotate('a - FFMC', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=10)
    ax1a.set_ylabel('Latitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')
    
    a_plot = ax1a.imshow(FFMC[tstep,:,:], vmin = 0, vmax = 100)
    plt.colorbar(a_plot,ax=ax1a)
    
    # Plot b -> the DMC
    ax1b = plt.subplot2grid((3,2),(1,0))
    ax1b.annotate('b - DMC', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=10)
    ax1b.set_ylabel('Latitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')
    
    b_plot = ax1b.imshow(DMC[tstep,:,:], vmin = 0, vmax = 200)
    plt.colorbar(b_plot,ax=ax1b)

    # Plot c -> the DC
    ax1c = plt.subplot2grid((3,2),(2,0))
    ax1c.annotate('c - DC', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=10)
    ax1c.set_ylabel('Latitude',fontsize=axis_size)
    ax1c.set_xlabel('Longitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    c_plot =ax1c.imshow(DC[tstep,:,:], vmin = 0, vmax = 2000)
    plt.colorbar(c_plot,ax=ax1c)
    
    # Plot d -> the ISI
    ax1d = plt.subplot2grid((3,2),(0,1))
    ax1d.annotate('d - ISI', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=10)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    d_plot = ax1d.imshow(ISI[tstep,:,:], vmin = 0, vmax = np.nanmax(ISI))
    plt.colorbar(d_plot,ax=ax1d)
    
    # Plot e -> the BUI
    ax1e = plt.subplot2grid((3,2),(1,1))
    ax1e.annotate('e - BUI', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=10)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    e_plot = ax1e.imshow(BUI[tstep,:,:], vmin = 0, vmax = np.nanmax(BUI))#, vmin = 0, vmax = 200)
    plt.colorbar(e_plot,ax=ax1e)
    
    # Plot f -> the FWI
    ax1f = plt.subplot2grid((3,2),(2,1))
    ax1f.annotate('f - FWI', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=10)
    ax1f.set_xlabel('Longitude',fontsize=axis_size)
    plt.gca().set_aspect('equal', adjustable='box-forced')

    f_plot =ax1f.imshow(FWI[tstep,:,:], vmin = 0, vmax = 40)
    plt.colorbar(f_plot,ax=ax1f)
    
    plt.show()
    return 0


# Plot time series for all FWI indices at specified timestep
def plot_FWI_indices_time_series_for_pixel(T,P,H,W,FFMC,DMC,DC,ISI,BUI,FWI,dates,start_tstep,end_tstep,row,col):
    if start_tstep < 0:
        start_tstep = 0
    if end_tstep > FWI.size:
        end_tstep = FWI.size
        
    plt.figure(2, facecolor='White',figsize=[8,12])
    # plot a -> relative humidity & mean temperature
    ax_a1 = host_subplot(511, axes_class=AA.Axes)
    ax_a2 = ax_a1.twinx()
    ax_a1.annotate('a - relative humidity & air temperature', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)
    ax_a2.set_ylabel('air temperature / $^o$C',fontsize=axis_size, color = colour[2])
    ax_a1.set_ylabel('relative humidity / %',fontsize=axis_size,color=colour[1])

    ax_a1.plot(dates[start_tstep:end_tstep],H[start_tstep:end_tstep,row,col],'-',colour=colour[1])
    ax_a2.plot(dates[start_tstep:end_tstep],T[start_tstep:end_tstep,row,col],'-',colour=colour[2])
    
    # plot b -> precipitation & wind speed
    ax_b1 = host_subplot(512, axes_class=AA.Axes)
    ax_b2 = ax_b1.twinx()
    ax_b1.annotate('b - precipitation & wind speed', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)
    ax_b1.set_ylabel('precipitation / mm',fontsize=axis_size, color = colour[1])
    ax_b2.set_ylabel('wind speed / m.s$^{-1}$',fontsize=axis_size,color=colour[2])
               
    ax_b1.plot(dates[start_tstep:end_tstep],P[start_tstep:end_tstep,row,col],'-',colour=colour[1])
    ax_b2.plot(dates[start_tstep:end_tstep],W[start_tstep:end_tstep,row,col],'-',colour=colour[2])
    
    # plot c -> the FFMC, DMC and DC
    ax_c1 = host_subplot(513, axes_class=AA.Axes)
    ax_c2 = ax_c1.twinx()
    ax_c3 = ax_c1.twinx()
    offset = 60
    new_fixed_axis = ax_c3.get_grid_helper().new_fixed_axis
    ax_c3.axis["right"] = new_fixed_axis(loc='right',axes = ax_c3, offset = (offset,0))
    ax_c3.axis["right"].toggle(all=True)
    ax_c1.annotate('c - FFMC, DMC, DC', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)
    ax_c1.set_ylabel('FFMC',fontsize=axis_size, color = colour[1])
    ax_c2.set_ylabel('DMC',fontsize=axis_size,color=colour[2])
    ax_c3.set_ylabel('DC',fontsize=axis_size,color=colour[0])
               
    ax_c1.plot(dates[start_tstep:end_tstep],FFMC[start_tstep:end_tstep,row,col],'-',colour=colour[1])
    ax_c2.plot(dates[start_tstep:end_tstep],DMC[start_tstep:end_tstep,row,col],'-',colour=colour[2])
    ax_c3.plot(dates[start_tstep:end_tstep],DC[start_tstep:end_tstep,row,col],'-',colour=colour[0])

    # plot d -> the BUI & ISI
    ax_d1 = host_subplot(514, axes_class=AA.Axes)
    ax_d2 = ax_d1.twinx()
    ax_d1.annotate('d -ISI & BUI', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)
    ax_d1.set_ylabel('ISI',fontsize=axis_size, color = colour[1])
    ax_d2.set_ylabel('BUI',fontsize=axis_size,color=colour[2])
               
    ax_d1.plot(dates[start_tstep:end_tstep],ISI[start_tstep:end_tstep,row,col],'-',colour=colour[1])
    ax_d2.plot(dates[start_tstep:end_tstep],BUI[start_tstep:end_tstep,row,col],'-',colour=colour[2])

    # plot e -> the FWI
    ax_e1 = host_subplot(515, axes_class=AA.Axes)
    ax_e1.annotate('e -FWI', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)
    ax_e1.set_ylabel('FWI',fontsize=axis_size, color = colour[2])
    ax_e1.set_xlable('timestep / days',fontsize=axis_size)

    ax_e1.plot(dates[start_tstep:end_tstep],ISI[start_tstep:end_tstep,row,col],'-',colour=colour[2])

    
    # plot f -> the burned area (tbc)
    plt.tight_layout()
    plt.show()

    
    plt.show()
    return 0
