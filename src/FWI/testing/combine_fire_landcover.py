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

start_year = 2003
end_year = 2008
year = np.arange(start_year,end_year+1)
Nyy = year.size
suffix = 'col'

# Bounding box for Colombia
W = -79.875
E = -65.875
N = 12.875
S = -4.875

# load MODIS fire data
print "loading MODIS burned area data"
month,lat_MODIS,lon_MODIS,burnday,unc = modis.load_MODIS_monthly_fire(start_year,end_year,suffix,clip=True,N=N,S=S,E=E,W=W)

"""
# Next regrid the landcover data to match the MODIS data.
# - generate the grid specification file
gridspec = open('MODIS_gridspec_colombia','w')
gridspec.write('gridtype = lonlat\n')
gridspec.write('xsize = %i\n' % lon.size)
gridspec.write('ysize = %i\n' % lat.size)
gridspec.write('xfirst = %f\n' % lon.min())
gridspec.write('xinc = %f\n' % (lon[1]-lon[0]))
gridspec.write('yfirst = %f\n' % lat.max())
gridspec.write('yinc = %f\n' %  (lat[1]-lat[0]))
gridspec.close()
"""

# load landcover data
print "loading ESA CCI land cover data"
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
io.write_array_to_GeoTiff(lc,geoT_lc,'ESACCI_LC_col_temp')

# use gdalwarp to resample the landcover grid using nearest neighbour
print "resampling land cover grid to MODIS grid"
dY_modis = lat_MODIS[1]-lat_MODIS[0]
dX_modis = lon_MODIS[1]-lon_MODIS[0]
N_target = lat_MODIS.max()+np.abs(dY_modis)/2.
S_target =lat_MODIS.min()-np.abs(dY_modis)/2.
W_target =lon_MODIS.min()-np.abs(dX_modis)/2.
E_target = lon_MODIS.max()+np.abs(dX_modis)/2.
os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f %f -r mode ESACCI_LC_col_temp.tif ESACCI_LC_col_MODISgrid.tif" % (W_target,S_target,E_target,N_target,dX_modis,dY_modis))
os.system("rm ESACCI_LC_col_temp.tif")

lc=None
# load in new landcover raster
lc, geoT_lc, coord_sys_lc = io.load_raster_and_georeferencing('ESACCI_LC_col_MODISgrid.tif')
lc=np.swapaxes(lc,1,2)
lc=np.swapaxes(lc,0,1)
lc=lc.astype(int)

# For MODIS burned area, aggregate to individual fires using a connected components algorithm
# - layers created:
#   - fires labelled with unique ID (3D array -month,lat,lon- corresponding with MODIS burnday)
#   - Nfires: number of individual fires in the month 
#   - Npixels: number of fire pixels in the month
#   - ignition_day: list of numpy vectors, one for each month, containing the earliest date for respective fires in each month
#                   Note that ignition day gives the Julian Day for that year
#   - ignition_lc: list of numpy vectors, one for each month, indicating the landcover for which ignition date was recorded
#   - majority_lc: list of numpy vectors, one for each month, the principal land cover type affected by each fire
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

        # retrieve the fire-affected pixels for this month
        fires_iter = fires[ii][fires[ii]>0]
        burnday_iter=burnday[ii][fires[ii]>0]
        lc_iter=lc[yy][fires[ii]>0]

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
            
        # increment index
        ii+=1
        # increment total number of fires
        Nfires_total+=fire_labels.size

        if ii == Ntsteps:
            break
    if ii == Ntsteps:
        break

# finally, keep track of the observed vs. non-observed pixels in a given month 
observed_pixels = np.zeros(burnday.shape)
observed_pixels[np.isfinite(burnday)]==1.

#===================================================================================
# Now we have the fire data processed, we need to load in the met data
# Once the met data is loaded in, the above fire info will be regridded to
# ascertain the following:
# (i) The number of fires occurring on each day of the year
# (ii) The number of pixels for which we have observations for each day of the year
# (iii) The burned area for each day of the year

#-----------------------------------------------------------------------------------
# First, load in the meteorological data
print "Loading ERA Interim"
# - relative humidity in %
# - air temperature in oC - this should ideally be noontime temperature (~peak temperature)
# - wind speed in m/s
# - pptn in mm (need to convert from metres)
# - effective day length
path2met = '/disk/scratch/local.2/dmilodow/ERAinterim/source_files/0.25deg_Colombia/'
start_month=1
end_month = 12
date,lat_ERA,lon_ERA,rh = era.calculate_rh_daily(path2met,start_month,start_year,end_month,end_year)
#temp1,temp2,temp3,wind = era.calculate_wind_speed_daily(path2met,start_month,start_year,end_month,end_year)
#dates_prcp,temp2,temp3,prcp = era.load_ERAinterim_daily(path2met,'prcp',start_month,start_year,end_month,end_year)
#temp,lat,lon,mx2t = era.load_ERAinterim_daily(path2met,'mx2t',start_month,start_year,end_month,end_year)
#date,lat,lon,mn2t = era.load_ERAinterim_daily(path2met,'mn2t',start_month,start_year,end_month,end_year)
#temp1,temp2,temp3,psurf = era.load_ERAinterim_daily(path2met,'psurf',start_month,start_year,end_month,end_year)
#temp1,temp2,temp3,ssrd = era.load_ERAinterim_daily(path2met,'ssrd',start_month,start_year,end_month,end_year)
#mx2t=mx2t[:rh.shape[0],:,:]
#mn2t=mn2t[:rh.shape[0],:,:]
#t2m=(mx2t+mn2t)/2.
#prcp*=1000
#prcp[prcp<0]=0

# convert pressure to kPa
#psurf/=1000.
# convert ssrd from Jm-2d-1 to MJm-2d-1
#ssrd/=10**6

# Mask out oceans so that land areas are only considered
bm = Basemap()
land_mask = np.zeros((lat.size,lon.size))*np.nan
for ii in range(0,lat.size):
    for jj in range(0,lon.size):
        if bm.is_land(lon[jj],lat[ii]):
            land_mask[ii,jj] = 1

# now regrid the MODIS data to the resolution of ERA interim

    
