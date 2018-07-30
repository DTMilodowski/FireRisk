import numpy as np


# A fuzzy flood fill algorithm to separate merged fires
# input args:
# - burnday the burned area array giving date burned
# - lim the threshold number of days for fire to be included
def separate_fires(burnday,lim=8):
    # get rows and cols of fire-affected pixels
    rows,cols = np.where(burnday>0)
    n = rows.size
    # setup host array for segmented fires
    fires = np.zeros(burnday.shape)
    # Need an updating tag number for each fire
    tag = 1
    # Now loop through the fire pixels
    for ff in range(0,n):
        # if this fire pixel has not yet been assigned to a fire, start flood fill
        if fires[rows[ff],cols[ff]]==0:
            floodFill(burnday,fires,rows[ff],cols[ff],tag,lim)
            # update fire tag number for next fire
            tag+=1
        else:
            continue
    return fires

# Recursive flood fill function
# vals is the burned area dates
# patch is an array recording the progress of the flood fill algorithm (i.e. to ensure that
# once a pixel is assigned, it does not get revisited, thus avoiding infinite recursion)
# i is the test row and col
# j is the test row and col
# tag is the code flagging included pixels
# lim is the threshold below which neighbouring pixels are considered part of the same object
# ref is the reference against which to test the current pixel value. If not specified, then
# this is set as the value of the root pixel (i.e. the start of the recursive sequence).
def floodFill(vals,patch,i,j,tag,lim, ref):
    
    # The recursive algorithm. Starting at row i and column j, changes any adjacent
    # cells in the 4-pixel neighbourhood (left,up,right,down) if the difference of
    # their values is within the specified limit
    nrows,ncols = vals.shape
    if ref==None:
        ref=vals[i,j]
    
    if np.abs(vals[i,j]-ref) <= lim:
        # Within specified fuzzy limit, so update
        patch[i,j]=tag
        ref=vals[i,j]

        # Recursive calls. Make a recursive call as long as we are not on the boundary
        if j > 0: # left
            # also check that neighbour pixel has not already been assigned
            if patch[i,j-1]!=tag:
                floodFill(vals, patch, i, j-1, tag, lim,ref)
        if i > 0: # up
            if patch[i-1,j]!=tag:
                floodFill(vals, patch,i-1, j, tag, lim,ref)
        if j < ncols-1: # right
            if patch[i,j+1]!=tag:
                floodFill(vals, patch,i, j+1, tag, lim,ref)
        if i < nrows-1: # down
            if patch[i+1,j]!=tag:
                floodFill(vals, patch,i+1, j, tag,lim, ref)
                
    else:
        # Base case, do nothing.
        return
