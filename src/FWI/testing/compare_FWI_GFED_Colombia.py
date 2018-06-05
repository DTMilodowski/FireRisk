#===============================================================================
# compare_FWI_GFED.py
#-------------------------------------------------------------------------------
# D. T. Milodowski, November 2017
#-------------------------------------------------------------------------------
# This routine tests the FWI components against GFED burned area data
#===============================================================================
import numpy as np

from sklearn.externals import joblib

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.ticker as plticker

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

sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src')
import data_io as io

sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/meteorology')
import load_ERAinterim as era

sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/fire/')
import load_GFED as GFED

sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/generic/')
import calculate_PET as pet
import resample_raster as resample

sys.path.append('/home/dmilodow/DataStore_DTM/FOREST2020/FireRisk/src/RandomForest')
import random_forest_cal_val as rf
import plot_random_forest_regression_results as plotrf


import load_GLEAM as gleam
import clip_raster as clip
import load_ESA_CCI_lc as cci


# Bounding box for Colombia
W = -79.875
E = -65.875
N = 12.875
S = -4.875

FWI_npyfile = 'Colombia_FWI.npz'

# Dates of interest
start_month = 1
start_year = 2000
end_month = 12
end_year = 2016

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
path2met = '/disk/scratch/local.2/dmilodow/ERAinterim/source_files/0.25deg_Colombia/'
temp1,temp2,temp3,rh = era.calculate_rh_daily(path2met,start_month,start_year,end_month,end_year)
temp1,temp2,temp3,wind = era.calculate_wind_speed_daily(path2met,start_month,start_year,end_month,end_year)
dates_prcp,temp2,temp3,prcp = era.load_ERAinterim_daily(path2met,'prcp',start_month,start_year,end_month,end_year)
temp,lat,lon,mx2t = era.load_ERAinterim_daily(path2met,'mx2t',start_month,start_year,end_month,end_year)
date,lat,lon,mn2t = era.load_ERAinterim_daily(path2met,'mn2t',start_month,start_year,end_month,end_year)
temp1,temp2,temp3,psurf = era.load_ERAinterim_daily(path2met,'psurf',start_month,start_year,end_month,end_year)
temp1,temp2,temp3,ssrd = era.load_ERAinterim_daily(path2met,'ssrd',start_month,start_year,end_month,end_year)
mx2t=mx2t[:rh.shape[0],:,:]
mn2t=mn2t[:rh.shape[0],:,:]
t2m=(mx2t+mn2t)/2.
date=date[:rh.shape[0]]
prcp*=1000
prcp[prcp<0]=0

# convert pressure to kPa
psurf/=1000.
# convert ssrd from Jm-2d-1 to MJm-2d-1
ssrd/=10**6

# Mask out oceans so that land areas are only considered
bm = Basemap()
land_mask = np.zeros((lat.size,lon.size))*np.nan
for ii in range(0,lat.size):
    for jj in range(0,lon.size):
        if bm.is_land(lon[jj],lat[ii]):
            land_mask[ii,jj] = 1

# create lat, long and month grids
latgrid_ERA = np.zeros(t2m.shape[1:])
latgrid = np.zeros(burned_area.shape[1:])
longrid = np.zeros(burned_area.shape[1:])
for ll in range(0,lon.size):
    latgrid_ERA[:,ll] = lat.copy()
for ll in range(0,lon_gfed.size):
    latgrid[:,ll] = lat_gfed.copy()

for ll in range(0,lat_gfed.size):
    longrid[ll,:] = lon_gfed.copy()
    
n_months = dates_gfed.size
month_array = np.zeros(burned_area.shape)
for mm in range(0,n_months):
    month_array[mm,:,:] = float(dates_gfed[mm]-dates_gfed[mm].astype('datetime64[Y]').astype('datetime64[M]'))
    
    
# calculate day lengths
Le = fwi.calculate_Le(latgrid_ERA,date)

# Now calculate the FWI
print "Calculating FWI"
# Default params
FFMC_default = 60.
DMC_default = 20.
DC_default = 200.

N_t = date.size

FFMC = np.zeros(t2m.shape)
DMC = np.zeros(t2m.shape)
DC = np.zeros(t2m.shape)
BUI = np.zeros(t2m.shape)
ISI = np.zeros(t2m.shape)
FWI = np.zeros(t2m.shape)

# spin up for one year
for tt in range(0,12):
    if (tt+1)%100 == 0:
        print "%i/%i" % (tt+1,N_t)
    if tt == 0:
        # calculate FFMC
        FFMC0=np.zeros(t2m.shape[1:])+FFMC_default
        FFMC0 = fwi.calculate_FFMC_array(rh[tt,:,:],t2m[tt,:,:],wind[tt,:,:],prcp[tt,:,:],FFMC0)
        # calculate DMC
        DMC0=np.zeros(t2m.shape[1:])+DMC_default
        DMC0 = fwi.calculate_DMC_array(rh[tt,:,:],t2m[tt,:,:],prcp[tt,:,:],Le[tt,:,:],DMC0)
        # calculate DC
        DC0=np.zeros(t2m.shape[1:])+DC_default
        DC0 = fwi.calculate_DC_array(t2m[tt,:,:],prcp[tt,:,:],DC0)

    else: 
        # calculate FFMC
        FFMC0 = fwi.calculate_FFMC_array(rh[tt,:,:],t2m[tt,:,:],wind[tt,:,:],prcp[tt,:,:],FFMC0)
        # calculate DMC
        DMC0 = fwi.calculate_DMC_array(rh[tt,:,:],t2m[tt,:,:],prcp[tt,:,:],Le[tt,:,:],DMC0)
        # calculate DC
        DC0 = fwi.calculate_DC_array(t2m[tt,:,:],prcp[tt,:,:],DC0)

# Now run through full time series
for tt in range(0,N_t):
    if (tt+1)%100 == 0:
        print "%i/%i" % (tt+1,N_t)
    if tt == 0:
        # calculate FFMC
        #FFMC0=np.zeros(t2m.shape[1:])+FFMC_default
        FFMC[tt,:,:] = fwi.calculate_FFMC_array(rh[tt,:,:],t2m[tt,:,:],wind[tt,:,:],prcp[tt,:,:],FFMC0)
        # calculate DMC
        #DMC0=np.zeros(t2m.shape[1:])+DMC_default
        DMC[tt,:,:] = fwi.calculate_DMC_array(rh[tt,:,:],t2m[tt,:,:],prcp[tt,:,:],Le[tt,:,:],DMC0)
        # calculate DC
        #DC0=np.zeros(t2m.shape[1:])+DC_default
        DC[tt,:,:] = fwi.calculate_DC_array(t2m[tt,:,:],prcp[tt,:,:],DC0)

    else: 
        # calculate FFMC
        FFMC[tt,:,:] = fwi.calculate_FFMC_array(rh[tt,:,:],t2m[tt,:,:],wind[tt,:,:],prcp[tt,:,:],FFMC[tt-1,:,:])
        # calculate DMC
        DMC[tt,:,:] = fwi.calculate_DMC_array(rh[tt,:,:],t2m[tt,:,:],prcp[tt,:,:],Le[tt,:,:],DMC[tt-1,:,:])
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
np.savez(FWI_npyfile,FFMC=FFMC,DMC=DMC,DC=DC,BUI=BUI,ISI=ISI,FWI=FWI)

# Now use random forest framework to assess the ability of these metrics
# to predict burned areas
FWI_load = np.load(FWI_npyfile)
FWI = FWI_load['FWI']
ISI = FWI_load['ISI']
BUI = FWI_load['BUI']
FFMC = FWI_load['FFMC']
DMC = FWI_load['DMC']
DC = FWI_load['DC']

FWI_month = date.astype('datetime64[M]')
n_months = dates_gfed.size

FWI_resample = np.zeros(burned_area.shape)
FFMC_resample = np.zeros(burned_area.shape)
DMC_resample = np.zeros(burned_area.shape)
DC_resample = np.zeros(burned_area.shape)
ISI_resample = np.zeros(burned_area.shape)
BUI_resample = np.zeros(burned_area.shape)

for mm in range(0,n_months):
    month_mask = FWI_month==dates_gfed[mm]
    FWI_resample[mm] = np.max(FWI[month_mask,:,:-1],axis=0)
    ISI_resample[mm] = np.max(ISI[month_mask,:,:-1],axis=0)
    BUI_resample[mm] = np.max(BUI[month_mask,:,:-1],axis=0)
    FFMC_resample[mm] = np.max(FFMC[month_mask,:,:-1],axis=0)
    DMC_resample[mm] = np.max(DMC[month_mask,:,:-1],axis=0)
    DC_resample[mm] = np.max(DC[month_mask,:,:-1],axis=0)
    """
    FWI_resample[mm] = resample.resample_nearest_neighbour(FWI_temp,lat,lon,lat_gfed,lon_gfed,mode='max')
    ISI_resample[mm] = resample.resample_nearest_neighbour(ISI_temp,lat,lon,lat_gfed,lon_gfed,mode='max')
    BUI_resample[mm] = resample.resample_nearest_neighbour(BUI_temp,lat,lon,lat_gfed,lon_gfed,mode='max')
    FFMC_resample[mm] = resample.resample_nearest_neighbour(FFMC_temp,lat,lon,lat_gfed,lon_gfed,mode='max')
    DMC_resample[mm] = resample.resample_nearest_neighbour(DMC_temp,lat,lon,lat_gfed,lon_gfed,mode='max')
    DC_resample[mm] = resample.resample_nearest_neighbour(DC_temp,lat,lon,lat_gfed,lon_gfed,mode='max')
    """
#=======================
# Load in Worldpop data
# population density
pop_file = '/home/dmilodow/DataStore_GCEL/WorldPop/Population/LAC_PPP_2015_adj_v2.tif'
pop_i, geoT, coord_sys = io.load_raster_and_georeferencing(pop_file,driver = 'GTiff')

# resample to resolution of GFED (mean)
rows_pop,cols_pop,bands = pop_i.shape
lon0 = geoT[0]
dlon = geoT[1]
lat0 = geoT[3]
dlat = geoT[5]
lon_pop = np.arange(lon0,lon0+float(cols_pop)*dlon-dlon/2,dlon)+dlon/2.
lat_pop = np.arange(lat0,lat0+float(rows_pop)*dlat-dlat/2.,dlat)+dlat/2.
pop_i[pop_i<0]=np.nan
pop_i,lat_pop,lon_pop=clip.clip_raster(pop_i,lat_pop,lon_pop,N,S,E,W)
pop = resample.resample_nearest_neighbour(pop_i[:,:,0],lat_pop,lon_pop,lat_gfed,lon_gfed,mode='mean')

# Travel time (access)
access_file = '/home/dmilodow/DataStore_GCEL/WorldPop/Mosaics_1km/mastergrid_traveltime50k_1km.tif'
access_i, geoT, coord_sys = io.load_raster_and_georeferencing(access_file,driver = 'GTiff')
rows_acc,cols_acc,bands = access_i.shape
lon0 = geoT[0]
dlon = geoT[1]
lat0 = geoT[3]
dlat = geoT[5]
lon_acc = np.arange(lon0,lon0+float(cols_acc)*dlon-dlon/2,dlon)+dlon/2.
lat_acc = np.arange(lat0,lat0+float(rows_acc)*dlat-dlat/2.,dlat)+dlat/2.
access_i,lat_acc,lon_acc=clip.clip_raster(access_i,lat_acc,lon_acc,N,S,E,W)
access = resample.resample_nearest_neighbour(access_i[:,:,0],lat_acc,lon_acc,lat_gfed,lon_gfed,mode='mean')

##Load GLEAM soil moisture
dateGLEAM, latGLEAM,lonGLEAM,SMsurf = gleam.load_GLEAM_daily('SMsurf',2000,2016)
SMsurf,latGLEAM,lonGLEAM=clip.clip_raster_ts(SMsurf,latGLEAM,lonGLEAM,N,S,E,W)
dateGLEAM, latGLEAM,lonGLEAM,SMroot = gleam.load_GLEAM_daily('SMroot',2000,2016)
SMroot,latGLEAM,lonGLEAM=clip.clip_raster_ts(SMroot,latGLEAM,lonGLEAM,N,S,E,W)

# resample GLEAM to monthly
SMsurf_resample = np.zeros(burned_area.shape)
SMroot_resample = np.zeros(burned_area.shape)
monthGLEAM = dateGLEAM.astype('datetime64[M]')
n_months = dates_gfed.size
for mm in range(0,n_months):
    month_mask = monthGLEAM==dates_gfed[mm]
    SMsurf_resample[mm] = np.max(SMsurf[month_mask],axis=0)
    SMroot_resample[mm] = np.max(SMroot[month_mask],axis=0)

# Load AGB raster
agb_file = '/home/dmilodow/DataStore_GCEL/AGB/avitabile/Avitabile_AGB_Map/Avitabile_AGB_Map.tif'
agb_i, geoT, coord_sys = io.load_raster_and_georeferencing(agb_file,driver = 'GTiff')
rows_agb,cols_agb,bands = agb_i.shape
lon0 = geoT[0]
dlon = geoT[1]
lat0 = geoT[3]
dlat = geoT[5]
lon_agb = np.arange(lon0,lon0+float(cols_agb)*dlon-dlon/2,dlon)+dlon/2.
lat_agb = np.arange(lat0,lat0+float(rows_agb)*dlat-dlat/2.,dlat)+dlat/2.
agb_i,lat_agb,lon_agb=clip.clip_raster(agb_i,lat_agb,lon_agb,N,S,E,W)
agb = resample.resample_nearest_neighbour(agb_i[:,:,0],lat_agb,lon_agb,lat_gfed,lon_gfed,mode='mean')


# Load Land cover raster
lat_lc, lon_lc, landcover = cci.load_landcover(2015)
landcover,lat_lc,lon_lc=clip.clip_raster(landcover,lat_lc,lon_lc,N,S,E,W)
landcover = cci.aggregate_classes(landcover)

# create subsidiary data layers to aggregate further. In this instance, we want:
# - land fraction
landcover = landcover.astype('float')
landcover[landcover==0]=np.nan
land = np.ones(landcover.shape,dtype='float')*(landcover<10)
land = resample.resample_nearest_neighbour(land,lat_lc,lon_lc,lat_gfed,lon_gfed,mode='mean')
# - forest fraction
forest = np.ones(landcover.shape,dtype='float')*(landcover==2)
forest = resample.resample_nearest_neighbour(forest,lat_lc,lon_lc,lat_gfed,lon_gfed,mode='mean')
# - agriculture fraction
agri = np.ones(landcover.shape,dtype='float')*(landcover==1)
agri = resample.resample_nearest_neighbour(agri,lat_lc,lon_lc,lat_gfed,lon_gfed,mode='mean')
# - urban fraction
urban = np.ones(landcover.shape,dtype='float')*(landcover==5) 
urban = resample.resample_nearest_neighbour(urban,lat_lc,lon_lc,lat_gfed,lon_gfed,mode='mean')                                 
# - grass fraction
grass = np.ones(landcover.shape,dtype='float')*(landcover==3) 
grass = resample.resample_nearest_neighbour(grass,lat_lc,lon_lc,lat_gfed,lon_gfed,mode='mean')                                
# - shrub fraction
shrub = np.ones(landcover.shape,dtype='float')*(landcover==6)  
shrub = resample.resample_nearest_neighbour(shrub,lat_lc,lon_lc,lat_gfed,lon_gfed,mode='mean')          
                                                        

# Set up next iteration of models
FWI_vars = {}
FWI_vars['FWI']=FWI_resample.copy()    
FWI_vars['BUI']=BUI_resample.copy()    
FWI_vars['ISI']=ISI_resample.copy()    
FWI_vars['FFMC']=FFMC_resample.copy()    
FWI_vars['DMC']=DMC_resample.copy()    
FWI_vars['DC']=DC_resample.copy()
FWI_vars['lat']=latgrid.copy()
FWI_vars['lon']=longrid.copy()
FWI_vars['pop']=pop.copy()
FWI_vars['access']=access.copy()
FWI_vars['agb']=agb.copy()
FWI_vars['land']=land.copy()
FWI_vars['forest']=forest.copy()
FWI_vars['agri']=agri.copy()
FWI_vars['urban']=urban.copy()
FWI_vars['grass']=grass.copy()
FWI_vars['shrub']=shrub.copy()

null_vars = {}
null_vars['month']=month_array.copy()    
null_vars['lat']=latgrid.copy()
null_vars['lon']=longrid.copy()
null_vars['pop']=pop.copy()
null_vars['access']=access.copy()
null_vars['agb']=agb.copy()
null_vars['land']=land.copy()
null_vars['forest']=forest.copy()
null_vars['agri']=agri.copy()
null_vars['urban']=urban.copy()
null_vars['grass']=grass.copy()
null_vars['shrub']=shrub.copy()

GLEAM_vars = {}
GLEAM_vars['lat']=latgrid.copy()
GLEAM_vars['lon']=longrid.copy()
GLEAM_vars['pop']=pop.copy()
GLEAM_vars['access']=access.copy()
GLEAM_vars['agb']=agb.copy()
GLEAM_vars['SMsurf']=SMsurf_resample.copy()
GLEAM_vars['SMroot']=SMroot_resample.copy()
GLEAM_vars['land']=land.copy()
GLEAM_vars['forest']=forest.copy()
GLEAM_vars['agri']=agri.copy()
GLEAM_vars['urban']=urban.copy()
GLEAM_vars['grass']=grass.copy()
GLEAM_vars['shrub']=shrub.copy()

# save driving data for future use
burned_area_file = 'burned_area_colombia'
FWI_predictors_file = 'fwi_predictors_colombia'
null_predictors_file = 'null_predictors_colombia'
GLEAM_predictors_file = 'GLEAM_predictors_colombia'
np.savez(burned_area_file,burned_area=burned_area)
np.savez(FWI_predictors_file,FWI_vars)
np.savez(null_predictors_file,null_vars)
np.savez(GLEAM_predictors_file,GLEAM_vars)
"""
# load drivers (if repeating)
burned_area = np.load(burned_area_file+'.npz')['burned_area']
FWI_vars = np.load(FWI_predictors_file+'.npz')['arr_0'][()]
null_vars = np.load(null_predictors_file+'.npz')['arr_0'][()]
GLEAM_vars = np.load(GLEAM_predictors_file+'.npz')['arr_0'][()]


"""
# FWI & static variables
FWI_matrix, FWI_burned_area, FWI_names= rf.construct_variables_matrices(FWI_vars,burned_area,time_series=True)
model_FWI, cal_FWI, val_FWI, importance_FWI = rf.random_forest_regression_model_calval(FWI_matrix,FWI_burned_area,n_trees_in_forest = 200)

# save rf model
filename = 'rf_burnedarea_FWI_colombia.sav'
joblib.dump(model_FWI, filename)

model_FWI = joblib.load(filename)
importance_FWI=model_FWI.feature_importances_

FWI_mod = rf.apply_random_forest_regression(FWI_matrix,model_FWI)

# Month & static variables (our null model)
null_matrix, null_burned_area, null_names = rf.construct_variables_matrices(null_vars,burned_area,time_series=True)
null, cal_null, val_null, importance_null = rf.random_forest_regression_model_calval(null_matrix,null_burned_area,n_trees_in_forest = 200)

filename = 'rf_burnedarea_null_colombia.sav'
joblib.dump(null, filename)
null = joblib.load(filename)
importance_null=null.feature_importances_

null_mod = rf.apply_random_forest_regression(null_matrix,null)

# GLEAM soil moisture & static variables
GLEAM_matrix, GLEAM_burned_area, GLEAM_names = rf.construct_variables_matrices(GLEAM_vars,burned_area,time_series=True)
GLEAM, cal_GLEAM, val_GLEAM, importance_GLEAM = rf.random_forest_regression_model_calval(GLEAM_matrix,GLEAM_burned_area,n_trees_in_forest = 200)

filename = 'rf_burnedarea_GLEAM_colombia.sav'
joblib.dump(GLEAM, filename)
GLEAM = joblib.load(filename)
importance_GLEAM=GLEAM.feature_importances_

GLEAM_mod = rf.apply_random_forest_regression(GLEAM_matrix,GLEAM)

# Plot up results
fig = plt.figure(1, facecolor='White',figsize=[10,12])
ax1a= plt.subplot2grid((3,5),(0,0),colspan=2)
ax1b= plt.subplot2grid((3,5),(0,2),colspan=3)
ax1c= plt.subplot2grid((3,5),(1,0),sharey=ax1a,sharex=ax1a,colspan=2)
ax1d= plt.subplot2grid((3,5),(1,2),sharey=ax1b,colspan=3)
ax1e= plt.subplot2grid((3,5),(2,0),sharey=ax1a,sharex=ax1a,colspan=2)
ax1f= plt.subplot2grid((3,5),(2,2),sharey=ax1b,colspan=3)

plotrf.plot_regression_model(ax1a,FWI_burned_area/10.**6,FWI_mod/10.**6, cmap = 'plasma', x_lab = 'observed burned area / km$^2$', y_lab = 'modelled burned area / km$^2$',units_str='',gridsize = (800,200))
plotrf.plot_importances(ax1b,FWI_names,importance_FWI,rank=True)

plotrf.plot_regression_model(ax1c,null_burned_area/10.**6,null_mod/10.**6, cmap = 'plasma', x_lab = 'observed burned area / km$^2$', y_lab = 'modelled burned area / km$^2$',units_str='',gridsize = (800,200))
plotrf.plot_importances(ax1d,null_names,importance_null,rank=True)

plotrf.plot_regression_model(ax1e,GLEAM_burned_area/10.**6,GLEAM_mod/10.**6, cmap = 'plasma', x_lab = 'observed burned area / km$^2$', y_lab = 'modelled burned area / km$^2$',units_str='',gridsize = (800,200))
plotrf.plot_importances(ax1f,GLEAM_names,importance_GLEAM,rank=True)

plt.tight_layout()
plt.savefig('RandomForest_comparison_of_fire_risk_colombia.png')
plt.show()


# Plot some general figures
# Figure 1: Annual average burned area estimates
fig = plt.figure(2, facecolor='White',figsize=[12,12]) 
loc_x = plticker.MultipleLocator(base=5)
loc_y = plticker.MultipleLocator(base=5)
#loc_cb = plticker.MultipleLocator(base=10) 

ax1a= plt.subplot2grid((2,5),(0,0),colspan=3)
ax1a.yaxis.set_major_locator(loc_y)
ax1a.xaxis.set_major_locator(loc_x)
ax1a.yaxis.set_major_locator(loc_y)
ax1a.xaxis.set_major_locator(loc_x)
ax1a.annotate('a - Mexico', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size, color='white')
ax1a.set_xlim(xmin=W-0.125,xmax=E+0.125)
ax1a.set_ylim(ymin=S-0.125,ymax=N+0.125)
im1a=ax1a.imshow(np.sum(burned_area,axis=0)/(12.*10**6),vmin=0,cmap='plasma',origin='upper',extent=[W,E,S,N])   
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
ax1b.set_xlim(xmin=W-0.125,xmax=E+0.125)
ax1b.set_ylim(ymin=S-0.125,ymax=N+0.125)
#im1b=ax1b.imshow(np.sum(burned_area,axis=0)/12.,vmin=0,cmap='plasma',origin='lower',extent=[W,E,S,N])   
#ax1b.axis('image')          
ax1b.yaxis.set_major_locator(loc_y)
ax1b.xaxis.set_major_locator(loc_x)                
for tick in ax1b.get_yticklabels():
    tick.set_rotation(90)


ax1c= plt.subplot2grid((2,5),(1,0),colspan=5)
ax1c.yaxis.set_major_locator(loc_y)
ax1c.xaxis.set_major_locator(loc_x)
ax1c.yaxis.set_major_locator(loc_y)
ax1c.xaxis.set_major_locator(loc_x)
ax1c.annotate('c - Indonesia', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)
ax1c.set_xlim(xmin=W-0.125,xmax=E+0.125)
ax1c.set_ylim(ymin=S-0.125,ymax=N+0.125)
#im1c=ax1c.imshow(np.sum(burned_area,axis=0)/12.,vmin=0,cmap='plasma',origin='lower',extent=[W,E,S,N])   
#ax1c.axis('image')          
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

plt.show()
