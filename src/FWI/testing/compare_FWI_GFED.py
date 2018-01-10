#===============================================================================
# compare_FWI_GFED.py
#-------------------------------------------------------------------------------
# D. T. Milodowski, November 2017
#-------------------------------------------------------------------------------
# This routine tests the FWI components against GFED burned area data
#===============================================================================
import numpy as np
from matplotlib import pyplot as plt
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
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.set_cmap(cmaps.inferno) # set inferno as default - a good fiery one :-)

sys.path.append('/home/dmilodow/DataStore_DTM/FOREST2020/FireRisk/src/FWI/')
import calculate_FWI as fwi
import plot_FWI as fwi_p

sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/meteorology')
import load_ERAinterim as era

sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/fire/')
import load_GFED as GFED


# Bounding box for Mexico
W = -118.15
E = -85.95
N = 33.075
S = 14

# Dates of interest
start_month = 1
start_year = 2000
end_month = 12
end_year = 2000

# First load in the GFED data
print 'Loading GFED'
path2gfed='/home/dmilodow/DataStore_GCEL/GFED4/0.25deg/monthly_nc/'
variable = 'BurnedArea'

dates_gfed, lat_gfed, lon_gfed, burned_area = GFED.load_GFED4_monthly(path2gfed,variable,start_month,start_year,end_month,end_year,N,S,E,W)

# Load in the met data
print "Loading ERA Interim"
# - relative humidity in %
# - air temperature in oC - this should ideally be noontime temperature (~peak temperature)
# - wind speed in m/s
# - pptn in mm (need to convert from metres)
# - effective day length
path2met = '/disk/scratch/local.2/dmilodow/ERAinterim/source_files/0.175deg_Mexico'
temp1,temp2,temp3,rh = era.calculate_rh_daily(path2met,start_month,start_year,end_month,end_year)
temp1,temp2,temp3,wind = era.calculate_wind_speed_daily(path2met,start_month,start_year,end_month,end_year)
dates_prcp,temp2,temp3,prcp = era.load_ERAinterim_daily(path2met,'prcp',start_month,start_year,end_month,end_year)
date,lat,lon,t2m = era.load_ERAinterim_daily(path2met,'mx2t',start_month,start_year,end_month,end_year)
t2m=t2m[:rh.shape[0],:,:]
date=date[:rh.shape[0]]
prcp*=1000
prcp[prcp<0]=0
# Mask out oceans so that land areas are only considered
bm = Basemap()
land_mask = np.zeros((lat.size,lon.size))*np.nan
for ii in range(0,lat.size):
    for jj in range(0,lon.size):
        if bm.is_land(lon[jj],lat[ii]):
            land_mask[ii,jj] = 1


# Now calculate the FWI
print "Calculating FWI"
# Default params
FFMC_default = 60.
DMC_default = 20.
DC_default = 200.
EffectiveDayLength = 10.

N_t = date.size

FFMC = np.zeros(t2m.shape)
DMC = np.zeros(t2m.shape)
DC = np.zeros(t2m.shape)
BUI = np.zeros(t2m.shape)
ISI = np.zeros(t2m.shape)
FWI = np.zeros(t2m.shape)

for tt in range(0,N_t):
    if (tt+1)%100 == 0:
        print "%i/%i" % (tt+1,N_t)
    if tt == 0:
        # calculate FFMC
        FFMC0=np.zeros(t2m.shape[1:])+FFMC_default
        FFMC[tt,:,:] = fwi.calculate_FFMC_array(rh[tt,:,:],t2m[tt,:,:],wind[tt,:,:],prcp[tt,:,:],FFMC0)
        # calculate DMC
        DMC0=np.zeros(t2m.shape[1:])+DMC_default
        DMC[tt,:,:] = fwi.calculate_DMC_array(rh[tt,:,:],t2m[tt,:,:],prcp[tt,:,:],EffectiveDayLength,DMC0)
        # calculate DC
        DC0=np.zeros(t2m.shape[1:])+DC_default
        DC[tt,:,:] = fwi.calculate_DC_array(t2m[tt,:,:],prcp[tt,:,:],DC0)

    else: 
        # calculate FFMC
        FFMC[tt,:,:] = fwi.calculate_FFMC_array(rh[tt,:,:],t2m[tt,:,:],wind[tt,:,:],prcp[tt,:,:],FFMC[tt-1,:,:])
        # calculate DMC
        DMC[tt,:,:] = fwi.calculate_DMC_array(rh[tt,:,:],t2m[tt,:,:],prcp[tt,:,:],EffectiveDayLength,DMC[tt-1,:,:])
        # calculate DC
        DC[tt,:,:] = fwi.calculate_DC_array(t2m[tt,:,:],prcp[tt,:,:],DC[tt-1,:,:])
        
    # Calculate BUI, ISI and FWI
    BUI[tt,:,:] = fwi.calculate_BUI_array(DMC[tt,:,:],DC[tt,:,:])
    ISI[tt,:,:]= fwi.calculate_ISI_array(FFMC[tt,:,:],wind[tt,:,:])
    FWI[tt,:,:] = fwi.calculate_FWI_array(ISI[tt,:,:],BUI[tt,:,:])
    
    # Apply land mask to layer
    FFMC[tt,:,:]*=land_mask
    DMC[tt,:,:]*=land_mask
    DC[tt,:,:]*=land_mask
    BUI[tt,:,:]*=land_mask
    ISI[tt,:,:]*=land_mask
    FWI[tt,:,:]*=land_mask

# save FWI information so we don't need to keep processing the met data
FWI_npyfile = 'Mexico_FWI.npz'
np.savez(FWI_npyfile,FFMC=FFMC,DMC=DMC,DC=DC,BUI=BUI,ISI=ISI,FWI=FWI)

# Resample FWI so that it matches GFED resolution
