import numpy as np

from sklearn.externals import joblib

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

import plot_map_trio as plt_maps

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

# First load in the burned area reference data
GFED_ba = {}
burned_area_file = 'burned_area_colombia'
GFED_ba['col'] = np.load(burned_area_file+'.npz')['burned_area']
burned_area_file = 'burned_area_indo'
GFED_ba['ind'] = np.load(burned_area_file+'.npz')['burned_area']
burned_area_file = 'burned_area_mexico'
GFED_ba['mex'] = np.load(burned_area_file+'.npz')['burned_area']

# Next load in the predictor variables for the random forest regression models
FWI_vars = {}
null_vars = {}
GLEAM_vars = {}

FWI_predictors_file = 'fwi_predictors_colombia'
null_predictors_file = 'null_predictors_colombia'
GLEAM_predictors_file = 'GLEAM_predictors_colombia'

FWI_vars['col'] = np.load(FWI_predictors_file+'.npz')['arr_0'][()]
null_vars['col'] = np.load(null_predictors_file+'.npz')['arr_0'][()]
GLEAM_vars['col'] = np.load(GLEAM_predictors_file+'.npz')['arr_0'][()]

FWI_predictors_file = 'fwi_predictors_indo'
null_predictors_file = 'null_predictors_indo'
GLEAM_predictors_file = 'GLEAM_predictors_indo'

FWI_vars['ind'] = np.load(FWI_predictors_file+'.npz')['arr_0'][()]
null_vars['ind'] = np.load(null_predictors_file+'.npz')['arr_0'][()]
GLEAM_vars['ind'] = np.load(GLEAM_predictors_file+'.npz')['arr_0'][()]

FWI_predictors_file = 'fwi_predictors_mexico'
null_predictors_file = 'null_predictors_mexico'
GLEAM_predictors_file = 'GLEAM_predictors_mexico'

FWI_vars['mex'] = np.load(FWI_predictors_file+'.npz')['arr_0'][()]
null_vars['mex'] = np.load(null_predictors_file+'.npz')['arr_0'][()]
GLEAM_vars['mex'] = np.load(GLEAM_predictors_file+'.npz')['arr_0'][()]

# Next load in the RFr models
model_FWI={}
model_null={}
model_GLEAM={}

filename = 'rf_burnedarea_FWI_colombia.sav'
model_FWI['col'] = joblib.load(filename)           
filename = 'rf_burnedarea_FWI_indo.sav'
model_FWI['ind'] = joblib.load(filename)                     
filename = 'rf_burnedarea_FWI_mexico.sav'
model_FWI['mex'] = joblib.load(filename)

FWI_vars['col']['keys']= ['urban', 'DMC', 'FWI', 'lon', 'DC', 'pop', 'agb', 'access', 'shrub', 'ISI', 'BUI', 'FFMC', 'forest', 'agri', 'lat', 'grass', 'land']
FWI_vars['ind']['keys']= ['urban', 'DMC', 'FWI', 'lon', 'DC', 'pop', 'agb', 'access', 'shrub', 'ISI', 'BUI', 'FFMC', 'forest', 'agri', 'lat','grass', 'land']
FWI_vars['mex']['keys']= ['urban', 'DMC', 'FWI', 'lon', 'DC', 'pop', 'agb', 'access', 'shrub', 'ISI', 'BUI', 'FFMC', 'forest', 'agri', 'lat', 'grass', 'land']

filename = 'rf_burnedarea_null_colombia.sav'
model_null['col'] = joblib.load(filename)          
filename = 'rf_burnedarea_null_indo.sav'            
model_null['ind'] = joblib.load(filename)           
filename = 'rf_burnedarea_null_mexico.sav'
model_null['mex'] = joblib.load(filename)

null_vars['col']['keys']= ['urban', 'land', 'shrub', 'lon', 'agb', 'pop', 'month', 'access', 'forest', 'agri', 'lat', 'grass']
null_vars['ind']['keys']= ['urban', 'land', 'shrub', 'lon', 'agb', 'pop', 'month', 'access', 'forest', 'agri', 'lat', 'grass']
null_vars['mex']['keys']= ['urban', 'land', 'shrub', 'lon', 'agb', 'pop', 'month', 'access', 'forest', 'agri', 'lat', 'grass']


filename = 'rf_burnedarea_GLEAM_colombia.sav'
model_GLEAM['col'] = joblib.load(filename)
filename = 'rf_burnedarea_GLEAM_indo.sav'
model_GLEAM['ind'] = joblib.load(filename)
filename = 'rf_burnedarea_GLEAM_mexico.sav'
model_GLEAM['mex'] = joblib.load(filename)   

GLEAM_vars['col']['keys']=['urban', 'land', 'shrub', 'lon', 'agb', 'SMsurf', 'access', 'SMroot', 'pop', 'agri', 'lat', 'grass', 'forest']
GLEAM_vars['ind']['keys']=['urban', 'land', 'shrub', 'lon', 'agb', 'SMsurf', 'access', 'SMroot', 'pop', 'agri', 'lat' ,'grass', 'forest'] 
GLEAM_vars['mex']['keys']=['urban', 'land', 'shrub', 'lon', 'agb', 'SMsurf', 'access', 'SMroot', 'pop', 'agri', 'lat', 'grass', 'forest'] 

FWI_ba_mod = {}
null_ba_mod = {}
GLEAM_ba_mod = {}
region = ['ind','col','mex']



start_month = 1
start_year = 2000
end_month = 12
end_year = 2016
FWI_month = np.arange(np.datetime64('2000-01-01'),np.datetime64('2017-01-01')).astype('datetime64[M]')
month = np.arange(np.datetime64('2000-01'),np.datetime64('2017-01'))
"""
for rr in range(0,2):
    burned_area = GFED_ba[region[rr]].copy()
    FWI_resample = np.zeros(burned_area.shape)
    FFMC_resample = np.zeros(burned_area.shape)
    DMC_resample = np.zeros(burned_area.shape)
    DC_resample = np.zeros(burned_area.shape)
    ISI_resample = np.zeros(burned_area.shape)
    BUI_resample = np.zeros(burned_area.shape)
    for mm in range(0,month.size):
        month_mask = FWI_month==month[mm]
        FWI_resample[mm] = np.max(FWI_vars[region[rr]]['FWI'][month_mask],axis=0)
        ISI_resample[mm] = np.max(FWI_vars[region[rr]]['ISI'][month_mask],axis=0)
        BUI_resample[mm] = np.max(FWI_vars[region[rr]]['BUI'][month_mask],axis=0)
        FFMC_resample[mm] = np.max(FWI_vars[region[rr]]['FFMC'][month_mask],axis=0)
        DMC_resample[mm] = np.max(FWI_vars[region[rr]]['DMC'][month_mask],axis=0)
        DC_resample[mm] = np.max(FWI_vars[region[rr]]['DC'][month_mask],axis=0)


    FWI_vars[region[rr]]['FWI'] = FWI_resample.copy()
    FWI_vars[region[rr]]['ISI'] = ISI_resample.copy()
    FWI_vars[region[rr]]['BUI'] = BUI_resample.copy()
    FWI_vars[region[rr]]['FFMC'] =FFMC_resample.copy()
    FWI_vars[region[rr]]['DMC'] = DMC_resample.copy()
    FWI_vars[region[rr]]['DC'] = DC_resample.copy()
                                 
"""

for rr in range(0,3):
    null_ba_mod[region[rr]] = rf.apply_random_forest_classifier_to_dictionary_input(null_vars[region[rr]],GFED_ba[region[rr]],model_null[region[rr]],time_series=True)
    GLEAM_ba_mod[region[rr]] = rf.apply_random_forest_classifier_to_dictionary_input(GLEAM_vars[region[rr]],GFED_ba[region[rr]],model_GLEAM[region[rr]],time_series=True)
                                 
    FWI_ba_mod[region[rr]] = rf.apply_random_forest_classifier_to_dictionary_input(FWI_vars[region[rr]],GFED_ba[region[rr]],model_FWI[region[rr]],time_series=True)

    null_ba_mod[region[rr]][np.isnan(FWI_ba_mod[region[rr]])]=np.nan
    GLEAM_ba_mod[region[rr]][np.isnan(FWI_ba_mod[region[rr]])]=np.nan



# Now we have modelled burned area, calculate r-squared and RMSE


# calculate r-squared along first dimension of 3D array (e.g. spatially structured
# time series)
def calculate_spatial_rsq(x,y):
    ssxx = np.sum((x-np.mean(x,axis=0))**2,axis=0)
    ssyy = np.sum((y-np.mean(y,axis=0))**2,axis=0)
    ssxy = np.sum( (x-np.mean(x,axis=0))*(y-np.mean(y,axis=0)),axis=0)
    rsq = (ssxy**2)/(ssxx*ssyy)
    return rsq

# calculate bias along first dimension of 3D array (e.g. spatially structured
# time series)
def calculate_spatial_bias(x,y):
    bias = np.mean((y-x),axis=0)
    return bias

# RMSE along first dimension of 3D array (e.g. spatially structured
# time series)
def calculate_spatial_RMSE(x,y):
    RMSE = np.sqrt(np.mean(((y-x)**2),axis=0))
    return RMSE

# calculate annual average estimates in sq. km
mean={}
rsq={}
bias={}
RMSE={}
for rr in range(0,3):
    rsq[region[rr]]={}
    bias[region[rr]]={}
    RMSE[region[rr]]={}
    mean[region[rr]]={}
    
    mod_FWI=FWI_ba_mod[region[rr]]
    mod_null=null_ba_mod[region[rr]]
    mod_GLEAM=GLEAM_ba_mod[region[rr]]
    obs=GFED_ba[region[rr]]

    rsq[region[rr]]['FWI']=calculate_spatial_rsq(obs,mod_FWI)
    rsq[region[rr]]['null']=calculate_spatial_rsq(obs,mod_null)
    rsq[region[rr]]['GLEAM']=calculate_spatial_rsq(obs,mod_GLEAM)

    bias[region[rr]]['FWI']=calculate_spatial_bias(obs,mod_FWI)*12/10.**6
    bias[region[rr]]['null']=calculate_spatial_bias(obs,mod_null)*12/10.**6
    bias[region[rr]]['GLEAM']=calculate_spatial_bias(obs,mod_GLEAM)*12/10.**6

    RMSE[region[rr]]['FWI']=calculate_spatial_RMSE(obs,mod_FWI)*12/10.**6
    RMSE[region[rr]]['null']=calculate_spatial_RMSE(obs,mod_null)*12/10.**6
    RMSE[region[rr]]['GLEAM']=calculate_spatial_RMSE(obs,mod_GLEAM)*12/10.**6

    mean[region[rr]]['FWI']=np.sum(mod_FWI,axis=0)/17/10.**6
    mean[region[rr]]['null']=np.sum(mod_null,axis=0)/17/10.**6
    mean[region[rr]]['GLEAM']=np.sum(mod_GLEAM,axis=0)/17/10.**6


# Plot simulated burned area maps
mod = ['FWI','null','GLEAM']
for i in range(0,3):
    print mod[i]
    print '\t mean'
    maps.plot_map_trio(3,'predicted_burned_area_%s.png' % mod[i],mean['mex'][mod[i]],mean['col'][mod[i]],mean['ind'][mod[i]],cbar_label='predicted annual burned area / km$^2$',vmin=0,vmax=40)
    print '\t rsq'
    maps.plot_map_trio(4,'rsq_%s.png' % mod[i],rsq['mex'][mod[i]],rsq['col'][mod[i]],rsq['ind'][mod[i]],cbar_label='local $R^2$',vmin=0,vmax=1)
    print '\t bias'
    maps.plot_map_trio(5,'bias_%s.png' % mod[i],bias['mex'][mod[i]],bias['col'][mod[i]],bias['ind'][mod[i]],cbar_label='average annual bias / km$^2$',cmap='RdYlBu_r',vmin=-40,vmax=40)
    print '\t rmse'
    maps.plot_map_trio(6,'RMSE_%s.png' % mod[i],RMSE['mex'][mod[i]],RMSE['col'][mod[i]],RMSE['ind'][mod[i]],cbar_label='average annual RMSE / km$^2$',vmin=0,vmax=40)

# Plot Mexico maps only
print '\t mean'
maps.plot_map_trio_mexico(3,'predicted_burned_area_mexico.png',mean['mex'],cbar_label='predicted annual burned area / km$^2$',vmin=0,vmax=40)
print '\t rsq'
maps.plot_map_trio_mexico(4,'rsq_mexico.png',rsq['mex'],cbar_label='local $R^2$',vmin=0,vmax=1)
print '\t bias'
maps.plot_map_trio_mexico(5,'bias_mexico.png',bias['mex'],cbar_label='average annual bias / km$^2$',cmap='RdYlBu_r',vmin=-40,vmax=40)
print '\t rmse'
maps.plot_map_trio_mexico(6,'RMSE_mexico.png',RMSE['mex'],cbar_label='average annual RMSE / km$^2$',vmin=0,vmax=40)



# now plot r-squared and bias against burned area for pixel
fig = plt.figure(figure_number, facecolor='White',figsize=[12,4]) 
    
ax1a= plt.subplot2grid((1,3),(0,0))
ax1b= plt.subplot2grid((1,3),(0,1))
ax1c= plt.subplot2grid((1,3),(0,2))
ax1a.annotate('a - Mexico', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)
ax1b.annotate('b - Colombia', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)
ax1c.annotate('c - Indonesia', xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=axis_size)

plt.savefig(save_name)