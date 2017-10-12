#==============================================================================
# FWI_driver.py
#
# David T. Milodowski
# September 2017
#
# This is a simple driver function to test out the FWI module. It loads in the
# meteorological data, spins up the moisture codes and then iterates through
# the timesteps. At a later point, this will be incorporated into a dedicated
# object
#==============================================================================
# standard libraries
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

import sys
sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/meteorology')

# own libraries
import calculate_FWI as fwi
import load_ERAinterim as era
import plot_FWI as fwi_p
#-------------------------------------------------------------------------------
# locate files
path2files = '/disk/scratch/local.2/dmilodow/ERAinterim/source_files/0.175deg_Mexico'

# Set up fire risk simulation params
start_month = 1
start_year = 2000
end_month = 12
end_year = 2016

# Default params
FFMC_default = 60.
DMC_default = 20.
DC_default = 200.
EffectiveDayLength = 10.

# Load in the met data
# - relative humidity in %
# - air temperature in oC - this should ideally be noontime temperature (~peak temperature)
# - wind speed in m/s
# - pptn in mm (need to convert from metres)
# - effective day length
temp1,temp2,temp3,rh = era.calculate_rh_daily(path2files,start_month,start_year,end_month,end_year)
temp1,temp2,temp3,wind = era.calculate_wind_speed_daily(path2files,start_month,start_year,end_month,end_year)
dates_prcp,temp2,temp3,prcp = era.load_ERAinterim_daily(path2files,'prcp',start_month,start_year,end_month,end_year)
date,lat,lon,t2m = era.load_ERAinterim_daily(path2files,'mx2t',start_month,start_year,end_month,end_year)
prcp*=1000
prcp[prcp<0]=0
# Mask out oceans so that land areas are only considered
bm = Basemap()
land_mask = np.zeros((lat.size,lon.size))*np.nan
for ii in range(0,lat.size):
    for jj in range(0,lon.size):
        if bm.is_land(lon[jj],lat[ii]):
            land_mask[ii,jj] = 1

# Now calculate FWI indices
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

# example plot
tstep = -1
fwi_p.plot_FWI_indices_for_tstep(FFMC,DMC,DC,ISI,BUI,FWI,tstep)

