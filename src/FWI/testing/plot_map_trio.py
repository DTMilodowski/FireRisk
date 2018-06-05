import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.ticker as plticker
import matplotlib.colors as mcolors

from matplotlib import rcParams
from mpl_toolkits.basemap import Basemap
# Set up some basiic parameters for the plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['arial']
rcParams['font.size'] = 8
rcParams['legend.numpoints'] = 1
axis_size = rcParams['font.size']+2

import sys

# Get perceptually uniform colourmaps
sys.path.append('/home/dmilodow/DataStore_DTM/FOREST2020/EOdata/EO_data_processing/src/plot_EO_data/colormap/')
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)
plt.set_cmap(cmaps.plasma) # set inferno as default - a good fiery one :-)

def plot_map_trio(figure_number,save_name,data_mex,data_col,data_ind,vmin=0,vmax=0,mask=False,mask_mex=np.empty([]),mask_col=np.empty([]),mask_ind=np.empty([]), cbar_label='',cmap='plasma',symmetric=False):
    # Bounding box for Mexico
    Wm = -118.125
    Em = -85.875
    Nm = 33.125
    Sm = 13.875
    
    # Bounding box for Colombia
    Wc = -79.875
    Ec = -65.875
    Nc = 12.875
    Sc = -4.875
    
    # Bounding box for Indonesia
    Wi = 92
    Ei = 154
    Ni = 8
    Si = -12
    
    # set vlims if unset
    if vmin == vmax:
        vmin = np.nanmin(np.array([np.nanmin(data_mex),np.nanmin(data_col),np.nanmin(data_ind)]))
        vmax = np.nanmax(np.array([np.nanmax(data_mex),np.nanmax(data_col),np.nanmax(data_ind)]))
        if symmetric:
            vmax = max(np.abs(vmin),np.abs(vmax))
            vmin = -vmax
        
    fig = plt.figure(figure_number, facecolor='White',figsize=[12,10]) 
    loc_x = plticker.MultipleLocator(base=5)
    loc_y = plticker.MultipleLocator(base=5)
    
    ax1a= plt.subplot2grid((2,5),(0,0),colspan=3)
    ax1a.yaxis.set_major_locator(loc_y)
    ax1a.xaxis.set_major_locator(loc_x)
    ax1a.yaxis.set_major_locator(loc_y)
    ax1a.xaxis.set_major_locator(loc_x)
    ax1a.annotate('a - Mexico', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=axis_size)
    ax1a.set_xlim(xmin=Wm-0.125,xmax=Em+0.125)
    ax1a.set_ylim(ymin=Sm-0.125,ymax=Nm+0.125)
    
    if mask:
        data_mex_mask = np.ma.masked_where(mask_mex,data_mex)
        im1a_=ax1a.imshow(data_mex,cmap='Greys',origin='upper',extent=[Wm,Em,Sm,Nm])   
        im1a=ax1a.imshow(data_mex_mask,vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wm,Em,Sm,Nm])
    else:
        im1a=ax1a.imshow(data_mex,vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wm,Em,Sm,Nm])
        
    ax1a.axis('image')          
    ax1a.yaxis.set_major_locator(loc_y)
    ax1a.xaxis.set_major_locator(loc_x)                
    for tick in ax1a.get_yticklabels():
        tick.set_rotation(90)
        
                       
    ax1b= plt.subplot2grid((2,5),(0,3),colspan=2)
    ax1b.yaxis.set_major_locator(loc_y)
    ax1b.xaxis.set_major_locator(loc_x)
    ax1b.yaxis.set_major_locator(loc_y)
    ax1b.xaxis.set_major_locator(loc_x)
    ax1b.annotate('b - Colombia', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)
    ax1b.set_xlim(xmin=Wc-0.125,xmax=Ec+0.125)
    ax1b.set_ylim(ymin=Sc-0.125,ymax=Nc+0.125)

    if mask:
        data_col_mask = np.ma.masked_where(mask_col,data_col)
        im1b_=ax1b.imshow(data_col,cmap='Greys',origin='upper',extent=[Wc,Ec,Sc,Nc])     
        im1b=ax1b.imshow(data_col_mask,vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wc,Ec,Sc,Nc])
    else:
        im1b=ax1b.imshow(data_col,vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wc,Ec,Sc,Nc])
        
    ax1b.axis('image')          
    ax1b.yaxis.set_major_locator(loc_y)
    ax1b.xaxis.set_major_locator(loc_x)                
    for tick in ax1b.get_yticklabels():
        tick.set_rotation(90)


    ax1c= plt.subplot2grid((2,5),(1,0),colspan=5)
    ax1c.yaxis.set_major_locator(loc_y)
    ax1c.xaxis.set_major_locator(loc_x)
    ax1c.yaxis.set_major_locator(loc_y)
    ax1c.xaxis.set_major_locator(loc_x)
    ax1c.annotate('c - Indonesia', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=axis_size)
    ax1c.set_xlim(xmin=Wi-0.125,xmax=Ei+0.125)
    ax1c.set_ylim(ymin=Si-0.125,ymax=Ni+0.125)

    if mask:
        data_ind_mask = np.ma.masked_where(mask_ind,data_ind)
        im1c_=ax1c.imshow(data_ind,cmap='Greys',origin='upper',extent=[Wi,Ei,Si,Ni])   
        im1c=ax1c.imshow(data_ind_mask,vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wi,Ei,Si,Ni])
    else:
        im1c=ax1c.imshow(data_ind,vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wi,Ei,Si,Ni])
        
    ax1c.axis('image')          
    ax1c.yaxis.set_major_locator(loc_y)
    ax1c.xaxis.set_major_locator(loc_x)                
    for tick in ax1c.get_yticklabels():
        tick.set_rotation(90)

                       
    divider1c = make_axes_locatable(ax1c)
    cax1c = divider1c.append_axes("right", size="5%", pad=0.05)
    cbar1c=plt.colorbar(im1a, cax=cax1c)
    cbar1c.ax.set_ylabel(cbar_label,fontsize = axis_size)
    cbar1c.solids.set_edgecolor("face")        
    plt.savefig(save_name)
    #plt.show()

def plot_map_trio_mexico(figure_number,save_name,data,vmin=0,vmax=0,mask=False,mask_mex=np.empty([]),cbar_label='',cmap='plasma',symmetric=False):
    # Bounding box for Mexico
    Wm = -118.125
    Em = -85.875
    Nm = 33.125
    Sm = 13.875

    # set vlims if unset
    if vmin == vmax:
        vmin = np.nanmin(np.array([np.nanmin(data['FWI']),np.nanmin(data['null']),np.nanmin(data['GLEAM'])]))
        vmax = np.nanmax(np.array([np.nanmax(data['FWI']),np.nanmax(data['null']),np.nanmax(data['GLEAM'])]))
        if symmetric:
            vmax = max(np.abs(vmin),np.abs(vmax))
            vmin = -vmax
        
    fig = plt.figure(figure_number, facecolor='White',figsize=[7,4]) 
    loc_x = plticker.MultipleLocator(base=5)
    loc_y = plticker.MultipleLocator(base=5)
    
    ax1a= plt.subplot2grid((1,3),(0,0))
    ax1a.yaxis.set_major_locator(loc_y)
    ax1a.xaxis.set_major_locator(loc_x)
    ax1a.yaxis.set_major_locator(loc_y)
    ax1a.xaxis.set_major_locator(loc_x)
    ax1a.annotate('a - FWI', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=axis_size)
    ax1a.set_xlim(xmin=Wm-0.125,xmax=Em+0.125)
    ax1a.set_ylim(ymin=Sm-0.125,ymax=Nm+0.125)
    
    if mask:
        data_mask = np.ma.masked_where(mask,data['FWI'])
        im1a_=ax1a.imshow(data['FWI'],cmap='Greys',origin='upper',extent=[Wm,Em,Sm,Nm])   
        im1a=ax1a.imshow(data_mask,vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wm,Em,Sm,Nm])
    else:
        im1a=ax1a.imshow(data['FWI'],vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wm,Em,Sm,Nm])
        
    ax1a.axis('image')          
    ax1a.yaxis.set_major_locator(loc_y)
    ax1a.xaxis.set_major_locator(loc_x)                
    for tick in ax1a.get_yticklabels():
        tick.set_rotation(90)
        
                       
    ax1b= plt.subplot2grid((1,3),(0,1))
    ax1b.yaxis.set_major_locator(loc_y)
    ax1b.xaxis.set_major_locator(loc_x)
    ax1b.yaxis.set_major_locator(loc_y)
    ax1b.xaxis.set_major_locator(loc_x)
    ax1b.annotate('b - "null"', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=axis_size)
    ax1b.set_xlim(xmin=Wm-0.125,xmax=Em+0.125)
    ax1b.set_ylim(ymin=Sm-0.125,ymax=Nm+0.125)

    if mask:
        data_mask = np.ma.masked_where(mask,data['null'])
        im1b_=ax1b.imshow(data['null'],cmap='Greys',origin='upper',extent=[Wm,Em,Sm,Nm])     
        im1b=ax1b.imshow(data_mask,vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wm,Em,Sm,Nm])
    else:
        im1b=ax1b.imshow(data['null'],vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wm,Em,Sm,Nm])
        
    ax1b.axis('image')          
    ax1b.yaxis.set_major_locator(loc_y)
    ax1b.xaxis.set_major_locator(loc_x)                
    for tick in ax1b.get_yticklabels():
        tick.set_rotation(90)


    ax1c= plt.subplot2grid((1,3),(0,2))
    ax1c.yaxis.set_major_locator(loc_y)
    ax1c.xaxis.set_major_locator(loc_x)
    ax1c.yaxis.set_major_locator(loc_y)
    ax1c.xaxis.set_major_locator(loc_x)
    ax1c.annotate('c - GLEAM', xy=(0.05,0.05), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='bottom', fontsize=axis_size)
    ax1c.set_xlim(xmin=Wm-0.125,xmax=Em+0.125)
    ax1c.set_ylim(ymin=Sm-0.125,ymax=Nm+0.125)

    if mask:
        data_mask = np.ma.masked_where(mask,data)
        im1c_=ax1c.imshow(data['GLEAM'],cmap='Greys',origin='upper',extent=[Wm,Em,Sm,Nm])   
        im1c=ax1c.imshow(data_mask,vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wm,Em,Sm,Nm])
    else:
        im1c=ax1c.imshow(data['GLEAM'],vmin=vmin,vmax=vmax,cmap=cmap,origin='upper',extent=[Wm,Em,Sm,Nm])
        
    ax1c.axis('image')          
    ax1c.yaxis.set_major_locator(loc_y)
    ax1c.xaxis.set_major_locator(loc_x)                
    for tick in ax1c.get_yticklabels():
        tick.set_rotation(90)

                       
    divider1c = make_axes_locatable(ax1c)
    cax1c = divider1c.append_axes("right", size="5%", pad=0.05)
    cbar1c=plt.colorbar(im1a, cax=cax1c)
    cbar1c.ax.set_ylabel(cbar_label,fontsize = axis_size)
    cbar1c.solids.set_edgecolor("face")        
    plt.savefig(save_name)
