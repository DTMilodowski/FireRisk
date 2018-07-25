# A set of cythonised functions to take a sparse array of dates (fire activity from
# MODIS) in a monthly lat-long tile and convert into time-lat-long voxel space.
# Original code template developed by Declan Valters; modified by David Milodowski
import cython
cimport numpy as np
import numpy as np

#from libc.math cimport isnan

#cdef float NAN 
#NAN = float("NaN")

# Or use a NoData vlue
#cdef int NoData = -9999

# set up ctypes for the variables in the function to allow it to cythonise nicely
#cdef int i = 0
#cdef int j = 0

#cdef int jmax
#cdef int imax
#cdef int kmax

# Best to turn off these decorators if you are debugging
# (Segfaults etc.)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

# This function takes in a two arguements:
#
# First, a three dimensional array (day,lat,long) that will host the voxelised data
#
# Second, a 2 dimensional array (lat-long) where:
#    0   = no fire
#    n>0 = day of month of fire,
#    NaN = no observations
#    n<0 = fire, but not source of ignition (i.e. it spread from neighbouring pixel).
# 
def voxelspace(np.ndarray[np.int64_t, ndim=3] voxelspace, np.ndarray[np.int64_t, ndim=2] one_month_data):
    cdef int i = 0			      	      	
    cdef int j = 0
    cdef int NAN = -9999		      
    cdef int jmax = one_month_data.shape[1]
    cdef int imax = one_month_data.shape[0]
    cdef int kmax = voxelspace.shape[0] # time dimension
    
    for j in range(jmax):
        for i in range(imax):
            # no observations
            if one_month_data[i,j]==NAN:
                voxelspace[0:kmax, i, j] = NAN  # NoData value
            # ignition
            elif  one_month_data[i,j] > 0:
                voxelspace[0:int(one_month_data[i,j]), i, j] = 0
                voxelspace[int(one_month_data[i,j])-1, i, j] = True
                voxelspace[int(one_month_data[i,j]):kmax, i, j] = NAN
            # expansion of fire (demarked by negative values)
            elif one_month_data[i,j] < 0:
                voxelspace[0:int(one_month_data[i,j])-1, i, j] = 0
                voxelspace[int(one_month_data[i,j])-1:kmax, i, j] = NAN            
            # no fire
            elif one_month_data[i,j] == 0:
                voxelspace[:, i, j] = False
    return voxelspace
