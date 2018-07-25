import numpy as np
import os
import sys
import datetime as dt

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# simple plot to show how cumulative number of fire affected pixels increases
# through time. This allows us to infer whether the fire record is likely to
# adequately capture the spatial distribution of fire-affected area (saturation)
# or not (continued increase). This is important for understanding the extent
# to which the regional fire hazard is adequately captured by the data.

def plot_cumulative_fire_affected_pixels(fignum,figname,month,pix):
    fig = plt.figure(fignum, facecolor='White',figsize=[4,4])
    ax = plt.subplot2grid((1,1),(0,0))
    ax.plot(month.astype(dt.datetime),pix/1000.)
    ax.set_ylabel('Cumulative fire affected pixels / 10$^3$')
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()
    return 0
