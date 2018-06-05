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

sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/fire/')
import load_GFED as GFED

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

# Dates of interest
start_month = 1
start_year = 2000
end_month = 12
end_year = 2016

# First load in the GFED data
print 'Loading GFED'
path2gfed='/home/dmilodow/DataStore_GCEL/GFED4/0.25deg/monthly_nc/'
variable = 'BurnedArea'

dates_gfed, lat_gfed_m, lon_gfed_m, burned_area_m = GFED.load_GFED4_monthly(path2gfed,variable,start_month,start_year,end_month,end_year,Nm,Sm,Em,Wm)

dates_gfed, lat_gfed_c, lon_gfed_c, burned_area_c = GFED.load_GFED4_monthly(path2gfed,variable,start_month,start_year,end_month,end_year,Nc,Sc,Ec,Wc)

dates_gfed, lat_gfed_i, lon_gfed_i, burned_area_i = GFED.load_GFED4_monthly(path2gfed,variable,start_month,start_year,end_month,end_year,Ni,Si,Ei,Wi)

max_annual = np.max((np.sum(burned_area_i,axis=0).max(),np.sum(burned_area_c,axis=0).max(),np.sum(burned_area_m,axis=0).max()))/(17.*10**6)

bm = Basemap()
m_mask = np.zeros((lat_gfed_m.size,lon_gfed_m.size))*np.nan
for ii in range(0,lat_gfed_m.size):
    for jj in range(0,lon_gfed_m.size):
        if bm.is_land(lon_gfed_m[jj],lat_gfed_m[ii]):
            m_mask[ii,jj] = 1

c_mask = np.zeros((lat_gfed_c.size,lon_gfed_c.size))*np.nan
for ii in range(0,lat_gfed_c.size):
    for jj in range(0,lon_gfed_c.size):
        if bm.is_land(lon_gfed_c[jj],lat_gfed_c[ii]):
            c_mask[ii,jj] = 1

i_mask = np.zeros((lat_gfed_i.size,lon_gfed_i.size))*np.nan
for ii in range(0,lat_gfed_i.size):
    for jj in range(0,lon_gfed_i.size):
        if bm.is_land(lon_gfed_i[jj],lat_gfed_i[ii]):
            i_mask[ii,jj] = 1

for mm in range(0,dates_gfed.size):
    burned_area_m[mm]*=m_mask
    burned_area_c[mm]*=c_mask
    burned_area_i[mm]*=i_mask


    
            
# Plot some general figures
# Figure 1: Annual average burned area estimates
fig = plt.figure(1, facecolor='White',figsize=[12,10]) 
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
im1a=ax1a.imshow(np.sum(burned_area_m,axis=0)/(17.*10**6),vmin=0,vmax=max_annual,cmap='plasma',origin='upper',extent=[Wm,Em,Sm,Nm])   
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
im1b=ax1b.imshow(np.sum(burned_area_c,axis=0)/(17.*10**6),vmin=0,vmax=max_annual,cmap='plasma',origin='upper',extent=[Wc,Ec,Sc,Nc])   
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
im1c=ax1c.imshow(np.sum(burned_area_i,axis=0)/(17*10**6.),vmin=0,vmax=max_annual,cmap='plasma',origin='upper',extent=[Wi,Ei,Si,Ni])   
ax1c.axis('image')          
ax1c.yaxis.set_major_locator(loc_y)
ax1c.xaxis.set_major_locator(loc_x)                
for tick in ax1c.get_yticklabels():
    tick.set_rotation(90)

                       
divider1c = make_axes_locatable(ax1c)
cax1c = divider1c.append_axes("right", size="5%", pad=0.05)
cbar1c=plt.colorbar(im1a, cax=cax1c)
cbar1c.ax.set_ylabel('averague annual burned area / km$^2$',fontsize = axis_size)
cbar1c.solids.set_edgecolor("face")             
#cbar1c.locator = loc_cb
#cbar1c.update_ticks()
plt.savefig('annual_burned_area_maps.png')
plt.show()


# now loop through years
burned_area_im=np.argsort(np.mean(np.reshape(burned_area_i,(17,12,lat_gfed_i.size,lon_gfed_i.size)),axis=0),axis=0)[-1]*i_mask
burned_area_im_mask = np.ma.masked_where(np.mean(burned_area_i,axis=0)<=0,burned_area_im)

burned_area_cm=np.argsort(np.mean(np.reshape(burned_area_c,(17,12,lat_gfed_c.size,lon_gfed_c.size)),axis=0),axis=0)[-1]*c_mask
burned_area_cm_mask=np.ma.masked_where(np.mean(burned_area_c,axis=0)<=0,burned_area_cm)

burned_area_mm=np.argsort(np.mean(np.reshape(burned_area_m,(17,12,lat_gfed_m.size,lon_gfed_m.size)),axis=0),axis=0)[-1]*m_mask
burned_area_mm_mask=np.ma.masked_where(np.mean(burned_area_m,axis=0)<=0,burned_area_mm)

cm1 = cmaps.viridis(np.linspace(0., 1, 128))
cm2 = cmaps.plasma(np.linspace(0., 1, 128))[::-1]
# combine them and build a new colormap
colors_cycle = np.vstack((cm1, cm2))
cyclic = mcolors.LinearSegmentedColormap.from_list('cyclic', colors_cycle)
plt.register_cmap(name='cyclic', cmap=cyclic)

fig = plt.figure(2, facecolor='White',figsize=[12,10]) 
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
im1a_=ax1a.imshow(burned_area_mm,cmap='Greys',vmin=11,vmax=13,origin='upper',extent=[Wm,Em,Sm,Nm])   
im1a=ax1a.imshow(burned_area_mm_mask,vmin=0,vmax=12,cmap='cyclic',origin='upper',extent=[Wm,Em,Sm,Nm]) 
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
im1b_=ax1b.imshow(burned_area_cm,cmap='Greys',vmin=11,vmax=13,origin='upper',extent=[Wc,Ec,Sc,Nc])     
im1b=ax1b.imshow(burned_area_cm_mask,vmin=0,vmax=12,cmap='cyclic',origin='upper',extent=[Wc,Ec,Sc,Nc])  
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
im1c_=ax1c.imshow(burned_area_im,cmap='Greys',vmin=11,vmax=13,origin='upper',extent=[Wi,Ei,Si,Ni])   
im1c=ax1c.imshow(burned_area_im_mask,vmin=0,vmax=12,cmap='cyclic',origin='upper',extent=[Wi,Ei,Si,Ni])  
ax1c.axis('image')          
ax1c.yaxis.set_major_locator(loc_y)
ax1c.xaxis.set_major_locator(loc_x)                
for tick in ax1c.get_yticklabels():
    tick.set_rotation(90)

                       
divider1c = make_axes_locatable(ax1c)
cax1c = divider1c.append_axes("right", size="5%", pad=0.05)
cbar1c=plt.colorbar(im1a, cax=cax1c,ticks=[0.5,11.5])
cbar1c.ax.set_ylabel('principal fire month',fontsize = axis_size)
cbar1c.solids.set_edgecolor("face")        
cbar1c.ax.set_yticklabels(['Jan', 'Dec'])
plt.savefig('principal_fire_month.png')
plt.show()
