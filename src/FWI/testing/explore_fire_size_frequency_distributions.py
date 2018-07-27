import numpy as np
import os
import sys
sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/')
sys.path.append('/home/dmilodow/DataStore_DTM/FOREST2020/FireRisk/src/generic/')
sys.path.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/meteorology')
import load_ERAinterim as era
import load_MODIS as modis
import load_ESA_CCI_lc as cci
import clip_raster as clip
import resample_raster as rs
import data_io as io
from scipy import ndimage

import plot_fire_statistics as pfire


start_year = 2001
end_year = 2015
year = np.arange(start_year,end_year+1)
Nyy = year.size
suffix = 'mex'

# Bounding box for Jalisco, Mexico
W = -105.75
E = -101.
N = 23.
S = 17.5

# load MODIS fire data
print "loading MODIS burned area data"
month,lat_MODIS,lon_MODIS,burnday,unc = modis.load_MODIS_monthly_fire(start_year,end_year,suffix,clip=True,N=N,S=S,E=E,W=W)

# load landcover data
print "loading ESA CCI land cover data"
""" commented out as I've already resampled the landcover data
landcover = []
for yy in range(0,Nyy):
    print start_year + yy
    lat_lc, lon_lc, lc_yy = cci.load_landcover(year[yy])
    lc_yy,lat_lc,lon_lc=clip.clip_raster(lc_yy,lat_lc,lon_lc,N,S,E,W)
    landcover.append(cci.aggregate_classes(lc_yy))

lc = np.asarray(landcover)
landcover=None # release memory

# write landcover arrays to geoTIFF
geoT_lc = np.zeros(6)
dY_lc = lat_lc[1]-lat_lc[0]
dX_lc = lon_lc[1]-lon_lc[0]
geoT_lc[3] = lat_lc[0]-dY_lc/2.
geoT_lc[5] = dY_lc
geoT_lc[0] = lon_lc[0]-dX_lc/2.
geoT_lc[1] = dX_lc
# get bbox for MODIS grid
lc=np.swapaxes(lc,0,1)
lc=np.swapaxes(lc,1,2)
io.write_array_to_GeoTiff(lc,geoT_lc,'ESACCI_LC_mex_temp')

# use gdalwarp to resample the landcover grid using nearest neighbour
print "resampling land cover grid to MODIS grid"
dY_modis = lat_MODIS[1]-lat_MODIS[0]
dX_modis = lon_MODIS[1]-lon_MODIS[0]
N_target = lat_MODIS.max()+np.abs(dY_modis)/2.
S_target =lat_MODIS.min()-np.abs(dY_modis)/2.
W_target =lon_MODIS.min()-np.abs(dX_modis)/2.
E_target = lon_MODIS.max()+np.abs(dX_modis)/2.
os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f %f -r mode ESACCI_LC_mex_temp.tif ESACCI_LC_jalisco_MODISgrid.tif" % (W_target,S_target,E_target,N_target,dX_modis,dY_modis))
os.system("rm ESACCI_LC_mex_temp.tif")

lc=None
"""
# load in new landcover raster
lc, geoT_lc, coord_sys_lc = io.load_raster_and_georeferencing('ESACCI_LC_jalisco_MODISgrid.tif')
lc=np.swapaxes(lc,1,2)
lc=np.swapaxes(lc,0,1)
lc=lc.astype(int)

# First up, lets plot the cumulative number of fire-affected pixels through time.
# This allows us to infer whether the fire record is likely to adequately capture
# the spatial distribution of fire-affected area (saturation) or not (continued
# increase). This is important for understanding the extent to which the regional
# fire hazard is adequately captured by the data.
burnday_collapsed = burnday.reshape(burnday.shape[0],burnday.shape[1]*burnday.shape[2])
affected_pixels_temp = np.zeros(burnday_collapsed.shape)
affected_pixels_forest = np.zeros(burnday_collapsed.shape)
affected_pixels_agri = np.zeros(burnday_collapsed.shape)
affected_pixels_other = np.zeros(burnday_collapsed.shape)
n_months = burnday.shape[0]
for mm in range(0, n_months):
    mask = burnday_collapsed[mm]>0
    affected_pixels_temp[mm:,mask] = 1
    affected_pixels_forest[mm:,np.all((mask,lc.ravel()==2),axis=0)] = 1
    affected_pixels_agri[mm:,np.all((mask,lc.ravel()==1),axis=0)] = 1
    affected_pixels_other[mm:,np.all((mask,lc.ravel()>2),axis=0)] = 1

affected_pixels={}
affected_pixels['all'] = affected_pixels_temp.sum(axis=1)
affected_pixels['forest'] = affected_pixels_forest.sum(axis=1)
affected_pixels['agri'] = affected_pixels_agri.sum(axis=1)
affected_pixels['other'] = affected_pixels_other.sum(axis=1)

month_ts = np.arange(np.datetime64(str(start_year)+'-01','M'),np.datetime64(str(end_year+1)+'-01','M'))
pfire.plot_cumulative_fire_affected_pixels(1,'cumulative_fire_affected_pixels_jalisco.png',month_ts,affected_pixels)


## Next up look at the population statistics
print "aggregate fire information"
fires = np.zeros(burnday.shape)
Nfires= np.zeros(burnday.shape[0],dtype='int')
Npixels=np.sum(np.sum(burnday>0,axis=2),axis=1)
ignition_day = []
ignition_lc = []
majority_lc = []
fire_pixels = []
Ntsteps = burnday.shape[0]
ii = 0
Nfires_total = 0
for yy in range(0,Nyy):
    print 'year',yy+1,'/',Nyy
    for mm in range(0,12): 
        
        # pixels that fall outside fires are labelled with a zero
        fire_mask = burnday[ii]>0
        
        # use connected components algorithm to identify discrete burns
        # Note that may need to do something more complex e.g. Archibald & Roy (2009)
        # to account for aggregation of large fires
        fires[ii], Nfires[ii] = ndimage.label(fire_mask)
        
        # this ensures that every fire has a unique ID
        fires[ii][fires[ii]>0]+=Nfires_total

        # get labels for each fire
        fire_labels=np.unique(fires[ii])
        fire_labels=fire_labels[fire_labels>0]

        # prepare the fire info arrays for this month
        ignition_day.append(np.zeros(Nfires[ii]))
        ignition_lc.append(np.zeros(Nfires[ii]))
        majority_lc.append(np.zeros(Nfires[ii]))
        fire_pixels.append(np.zeros(Nfires[ii]))

        # retrieve the fire-affected pixels for this month
        fires_iter = fires[ii][fires[ii]>0]
        burnday_iter=burnday[ii][fires[ii]>0]
        #lc_iter=lc[yy][fires[ii]>0]
        lc_iter=lc[0][fires[ii]>0]

        # loop through the fires, extracting the ignition day, ignition lc and majority lc
        for ff in range(0,Nfires[ii]):
            mask = fires_iter==fire_labels[ff]
            # pull out the ignition date
            ignition_day[ii][ff] = np.min(burnday_iter[mask])
            # then the landcover for the ignition site
            ignition_mask = burnday_iter[mask]==ignition_day[ii][ff]
            ignition_lc[ii][ff]=np.bincount(lc_iter[mask][ignition_mask]).argmax()
            
            # then the majority landcover of the fire
            majority_lc[ii][ff]=np.bincount(lc_iter[mask]).argmax()

            # finally the number of pixels associated with each fire
            fire_pixels[ii][ff] = mask.sum()
            
        # increment total number of fires
        Nfires_total+=Nfires[ii]
        # increment index
        ii+=1

        if ii == Ntsteps:
            break
    if ii == Ntsteps:
        break

#convert list of arrays to single 1D array
fires = np.concatenate(fire_pixels,axis=0)
lc_class = np.concatenate(majority_lc,axis=0)
pfire.plot_fire_size_frequency_distribution(2,'fire_size_distributions_by_landcover_jalisco.png',fires,lc_class)


# finally, keep track of the observed vs. non-observed pixels in a given month
observed_pixels={}
observed_pixels['all']= np.isfinite(burnday).sum(axis=2).sum(axis=1)
forest_mask = np.ma.masked_array(burnday,mask=np.ones(burnday.shape)*(lc[0]==2)[np.newaxis,:,:])
observed_pixels['forest']=np.all((np.isfinite(burnday),forest_mask.mask),axis=0).sum(axis=2).sum(axis=1)
agri_mask = np.ma.masked_array(burnday,mask=np.ones(burnday.shape)*(lc[0]==1)[np.newaxis,:,:])
observed_pixels['agri']=np.all((np.isfinite(burnday),agri_mask.mask),axis=0).sum(axis=2).sum(axis=1)
other_mask = np.ma.masked_array(burnday,mask=np.ones(burnday.shape)*(lc[0]>2)[np.newaxis,:,:])
observed_pixels['other']=np.all((np.isfinite(burnday),other_mask.mask),axis=0).sum(axis=2).sum(axis=1)

pfire.plot_time_varying_activity(3,'fire_activity_ts_jalisco.png',month,observed_pixels,fire_pixels,majority_lc)
