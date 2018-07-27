import numpy as np


# A fuzzy flood fill algorithm to separate merged fires
# input args:
# - BA the burned area array giving date burned
# - fires_i the initial discretised fire map
# - n_fires the number of fires (calculated when fires_i is calculated
# - lim the threshold number of days for fire to be included
def separate_fires(BA,fires_i,n_fires,lim=8):
    fires = np.zeros(BA.shape)
    tag = 1
    for ff in in range(0,n_fires):
        mask = fires_i==ff+1 # labels start at 1
        n_pix = mask.sum()
        if n_pix>1: # only need to do any checks if we have more than one fire
            assigned = np.zeros(n_pix)
            idx=np.where(fires_i==ff+1) # get row and col for each pixel in fire
            
            # continue until all pixels are assigned
            assigned[0]=1
            while assigned.sum()<n_pix:
                
                
                if assigned.sum==n_pix:
                    tag+=1
                    break
                else:
                    continue
            
        else:
            fires[mask]=tag
            tag+=1

# Recursive flood fill function
# vals is the burned area dates
# i is the test row and col
# j is the test row and col
# patch is the 
def floodFill(vals,patch,i,j,tag,lim, ref_day):
    # The recursive algorithm. Starting at x and y, changes any adjacent
    # characters that match oldChar to newChar.
    nrows,ncols = vals.shape

    if ref_day==None:
        ref_day=vals[i,j]
        
    if np.abs(vals[i,j]-ref_day) <= lim:
        # Within specified fuzzy limit, so update
        patch[i,j]=tag
        ref_day=vals[i,j]

        # Recursive calls. Make a recursive call as long as we are not on the boundary
        if j > 0: # left
            floodFill(vals, patch, i, j-1, tag, lim,ref_day)

        if i > 0: # up
            floodFill(vals, patch,i-1, j, tag, lim,ref_day)

        if j < ncols-1: # right
            floodFill(vals, patch,i, j+1, tag, lim,ref_day)

        if i < nrows-1: # down
            floodFill(vals, patch,i+1, j, tag,lim, ref_day)
    else:
        # Base case. If the current x, y character is not the oldChar,
        # then do nothing.
        return


    
flood_fill_burned_area():


def floodfill(x, y, oldColor, newColor):

    # assume surface is a 2D image and surface[x][y] is the color at x, y.

    theStack = [ (x, y) ]

    while len(theStack) > 0:

        x, y = theStack.pop()

        if surface[x][y] != oldColor:

            continue

        surface[x][y] = newColor

        theStack.append( (x + 1, y) )  # right

        theStack.append( (x - 1, y) )  # left

        theStack.append( (x, y + 1) )  # down

        theStack.append( (x, y - 1) )  # up
