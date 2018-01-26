#===============================================================================
# resample_raster.py
#-------------------------------------------------------------------------------
# D. T. Milodowski, January 2017
#-------------------------------------------------------------------------------
# This module contains some resampling routines
#===============================================================================
import numpy as np
# Nearest neighbour
def resample_nearest_neighbour(array,lat_i,lon_i,lat_n,lon_n,mode='mean'):
    rows_i = lat_i.size
    cols_i = lon_i.size
    rows_n = lat_n.size
    cols_n = lon_n.size
    
    closest_lon=np.zeros(cols_i)
    closest_lat=np.zeros(rows_i)

    regrid = np.zeros((rows_n,cols_n))*np.nan
    
    #assign closest point in regrid lat to orig    
    for ii,val in enumerate(lat_i):
        closest_lat[ii]=np.argsort(np.abs(val-lat_n))[0]
    for jj,val in enumerate(lon_i):
        closest_lon[jj]=np.argsort(np.abs(val-lon_n))[0]
        
    for ii,lat_ii in enumerate(lat_n):
        lat_mask = closest_lat == ii
        
        for jj,lon_jj in enumerate(lon_n):
            lon_mask = closest_lon==jj
            if mode == 'mean':
                if array[lat_mask,lon_mask].size>0:
                    regrid[ii,jj] = np.mean(array[lat_mask,lon_mask])
            elif mode == 'max':
                if array[lat_mask,lon_mask].size>0:
                    regrid[ii,jj] = np.nanmax(array[lat_mask,lon_mask])
            elif mode == 'min':
                if array[lat_mask,lon_mask].size>0:
                    regrid[ii,jj] = np.nanmin(array[lat_mask,lon_mask])
            elif mode == 'sum':
                if array[lat_mask,lon_mask].size>0:
                    temp = array[lat_mask,lon_mask]
                    regrid[ii,jj] = np.sum(temp[np.isfinite(temp)])
            else:
                print "resample mode not supported"
                
    return regrid
