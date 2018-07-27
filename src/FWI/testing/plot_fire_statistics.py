import numpy as np
import os
import sys
import datetime as dt

from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set()
import powerlaw

import sys
sys.path.append('/home/dmilodow/DataStore_DTM/FOREST2020/EOdata/EO_data_processing/src/plot_EO_data/colormap/')
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.set_cmap(cmaps.inferno)

# code to get trio of nice colourblind friendly colours 
cmap = cm.get_cmap('inferno')
scale = np.arange(0.,4.)
scale /=3.5
colour = cmap(scale)

# simple plot to show how cumulative number of fire affected pixels increases
# through time. This allows us to infer whether the fire record is likely to
# adequately capture the spatial distribution of fire-affected area (saturation)
# or not (continued increase). This is important for understanding the extent
# to which the regional fire hazard is adequately captured by the data.

def plot_cumulative_fire_affected_pixels(fignum,figname,month,pix):
    fig = plt.figure(fignum, facecolor='White',figsize=[4,4])
    ax = plt.subplot2grid((1,1),(0,0))
    ax.plot(month.astype(dt.datetime),pix/1000.)
    ax.set_ylabel('Cumulative fire affected pixels / 10$^3$')
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()
    return 0

# as above, except here we include land cover to separate out the fire activity
# pix is now a dictionary of cumulatively affected pixels with the dictionary
# key indicating the landcover type
def plot_cumulative_fire_affected_pixels_by_landcover(fignum,figname,month,pix):
    lc = pix.keys()
    n_lc = len(lc)

    fig = plt.figure(fignum, facecolor='White',figsize=[4,4])
    ax = plt.subplot2grid((1,1),(0,0))
    all_fires = np.zeros(month.size)
    for ll in range(0,n_lc):
        ax.plot(month.astype(dt.datetime),pix[lc[ll]]/1000.,label = lc[ll])
        all_fires+=pix[lc[ll]]
        
    ax.plot(month.astype(dt.datetime),pix[lc[ll]]/1000., colour = 'black', label = 'all')    
    ax.set_ylabel('Cumulative fire affected pixels / 10$^3$')
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()
    return 0

# Simple plot for size frequency distribution of a set of fires. The input arguements are
# as follows:
# - figure number
# - figure name (for saving)
# - vector of fire sizes
def plot_fire_size_frequency_distribution(fignum,figname,fires):
    
    # first get the power law relationship
    fit=powerlaw.Fit(fires)
    
    fig = plt.figure(fignum, facecolor='White',figsize=[5,5])
    ax = plt.subplot2grid((1,1),(0,0))
    fit.plot_pdf(ax=ax, label = 'observations')
    fit.power_law.plot_pdf(linestyle='--', ax=ax, label = 'model')
    ax.set_ylabel('p(X)')
    ax.set_xlabel('Fire size / pixels')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()
    return 0

# Simple plot for size frequency distribution of a set of fires. The input arguements are
# as follows:
# - figure number
# - figure name (for saving)
# - vector of fire sizes
# - vector of land cover classes
def plot_fire_size_frequency_distribution(fignum,figname,fires,lc_class):

    # first get the power law relationship
    fit_all=powerlaw.Fit(fires)
    fit_forest=powerlaw.Fit(fires[lc_class==2])
    fit_agri=powerlaw.Fit(fires[lc_class==1])
    fit_other=powerlaw.Fit(fires[lc_class>2])
    
    fig = plt.figure(fignum, facecolor='White',figsize=[4,4])
    ax = plt.subplot2grid((1,1),(0,0))
    
    fit_all.plot_pdf(ax=ax, label = 'all',c=colour[0])
    fit_all.power_law.plot_pdf(linestyle='--', ax=ax,c=colour[0])
    fit_agri.plot_pdf(ax=ax, label = 'agriculture',c=colour[1])
    fit_agri.power_law.plot_pdf(linestyle='--', ax=ax,c=colour[1])
    fit_forest.plot_pdf(ax=ax, label = 'forest',c=colour[2])
    fit_forest.power_law.plot_pdf(linestyle='--', ax=ax,c=colour[2])
    fit_other.plot_pdf(ax=ax, label = 'other',c=colour[3])
    fit_other.power_law.plot_pdf(linestyle='--', ax=ax,c=colour[3])
    
    ax.set_ylabel('p(X)')
    ax.set_xlabel('Fire size / pixels')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()
    return 0

# Simple plot to show how (i) observed area; (ii) frequency, and (iii)
# total area affected by fire varies through time (% pixels affected
# per month, coloured by % of pixels observed)
def plot_time_varying_activity(fignum,figname,month,observed_pixels,fire_pixels, landcover):

    t = month.astype(dt.datetime)
    
    areas={}
    areas['all']=np.zeros(month.size)
    areas['forest']=np.zeros(month.size)
    areas['agri']=np.zeros(month.size)
    areas['other']=np.zeros(month.size)
    events={}
    events['all']=np.zeros(month.size)
    events['forest']=np.zeros(month.size)
    events['agri']=np.zeros(month.size)
    events['other']=np.zeros(month.size)
    for mm in range(0,len(fire_pixels)):
        events['all'][mm]=fire_pixels[mm].size
        areas['all'][mm]=fire_pixels[mm].sum()
        events['forest'][mm]=fire_pixels[mm][landcover[mm]==2].size
        areas['forest'][mm]=fire_pixels[mm][landcover[mm]==2].sum()
        events['agri'][mm]=fire_pixels[mm][landcover[mm]==1].size
        areas['agri'][mm]=fire_pixels[mm][landcover[mm]==1].sum()
        events['other'][mm]=fire_pixels[mm][landcover[mm]>2].size
        areas['other'][mm]=fire_pixels[mm][landcover[mm]>2].sum()
        
    fig = plt.figure(fignum, facecolor='White',figsize=[8,8])
    ax1 = plt.subplot2grid((3,1),(0,0))
    ax1.plot(t,observed_pixels['all']/1000.,'-',c=colour[0])
    ax1.plot(t,observed_pixels['agri']/1000.,'-',c=colour[1])
    ax1.plot(t,observed_pixels['forest']/1000.,'-',c=colour[2])
    ax1.plot(t,observed_pixels['other']/1000.,'-',c=colour[3])
    ax1.set_ylim(ymin=0)
    

    ax2 = plt.subplot2grid((3,1),(1,0),sharex=ax1)
    ax2.plot(t,areas['all']/observed_pixels['all'].astype('float'),'-',c=colour[0],label='all')
    ax2.plot(t,areas['agri']/observed_pixels['agri'].astype('float'),'-',c=colour[1],label='agriculture')
    ax2.plot(t,areas['forest']/observed_pixels['forest'].astype('float'),'-',c=colour[2],label='forest')
    ax2.plot(t,areas['other']/observed_pixels['other'].astype('float'),'-',c=colour[3],label='other')
    ax2.set_ylim(ymin=0)

    
    ax3 = plt.subplot2grid((3,1),(2,0),sharex=ax1)
    ax3.plot(t,events['all']/observed_pixels['all'].astype('float'),'-',c=colour[0],label='all')
    ax3.plot(t,events['agri']/observed_pixels['agri'].astype('float'),'-',c=colour[1],label='agriculture')
    ax3.plot(t,events['forest']/observed_pixels['forest'].astype('float'),'-',c=colour[2],label='forest')
    ax3.plot(t,events['other']/observed_pixels['other'].astype('float'),'-',c=colour[3],label='other')
    ax3.set_ylim(ymin=0)

    
    ax1.set_ylabel('10$^3$ pixels observed')
    ax2.set_ylabel('area burned / %')
    ax3.set_ylabel('fire incidence rate / pixels$^{-1}$')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()
    return 0
