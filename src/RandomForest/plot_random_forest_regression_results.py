#========================================================================
# plot_random_forest_regression_results.py
#------------------------------------------------------------------------
# This package hosts a bunch of code to plot the results from random
# forest regression modelling.
# They provide the basic visualisation tools to evaulate the performance
# of random regression models.
#------------------------------------------------------------------------
# David T. Milodowski, March 2018

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from datetime import datetime

import sys

# Set up some basiic parameters for the plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['arial']
rcParams['font.size'] = 8
rcParams['legend.numpoints'] = 1
axis_size = rcParams['font.size']+2

# Get perceptually uniform colourmaps
sys.path.append('/home/dmilodow/DataStore_DTM/FOREST2020/EOdata/EO_data_processing/src/plot_EO_data/colormap/')
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)
plt.set_cmap(cmaps.inferno) # set inferno as default - a good fiery one :-)

# get default colours
cmap = cm.get_cmap('plasma')
scale = np.arange(0.,3.)
scale /=2.5
colour = cmap(scale)

# import some stats stuff
from statsmodels.stats.outliers_influence import summary_table
import statsmodels.api as sm


# Plot regression model
# Inputs
# - ax: an axis object
# - observations (Y)
# - model (X)
# - cmap: (optional) colour ramp used for hexbin plot (default is plasma)
# - gridsize: (optional) define resolution of hexgrid (default 100). Can take
#   (xres, yres) as input
# - x_lab, y_lab: labels for x and y axes respectively
# - units_str: the units for the model and obs
def plot_regression_model(ax,obs,mod, cmap = 'plasma',gridsize = 100., y_lab = 'model', x_lab = 'observations',units_str=''):

    # convert to 1D arrays
    mod = mod.ravel()
    obs = obs.ravel()
    # plot hexbin
    hb  = ax.hexbin(obs, mod, gridsize=gridsize, cmap=cmap, bins='log')    
    #cb = fig.colorbar(hb, ax=ax)
    #cb.set_label('counts')
    
    # now plot regression model and CI
    
    X = obs.copy()
    Y= mod.copy()
    X = sm.add_constant(X)

    model = sm.OLS(Y,X)
    results = model.fit()
    st, data, ss2 = summary_table(results, alpha=0.05)
    
    fittedvalues = data[:,2]
    predict_mean_se  = data[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    predict_ci_low, predict_ci_upp = data[:,6:8].T
    
    idx = np.argsort(fittedvalues)
    ax.plot(obs[idx],fittedvalues[idx],'-',color='white',label='Least Square Regression',lw=2)
    idx = np.argsort(predict_ci_low)
    ax.plot(obs[idx],predict_ci_low[idx],'--',color = 'white',lw=2,label='95% confidence interval')
    idx = np.argsort(predict_ci_upp)
    ax.plot(obs[idx],predict_ci_upp[idx],'--',color = 'white',lw=2)

    #x = np.ceil(max(obs.max(),fittedvalues.max()))
    mx = max(mod)
    ax.plot([0,mx],[0,mx],':',color='white', lw=1)

   
    ax.legend(loc='upper left')
    ax.set_xlabel(x_lab,fontsize=axis_size)
    ax.set_ylabel(y_lab,fontsize=axis_size)

    nse = 1-((Y-obs)**2).sum()/((obs-obs.mean())**2).sum()
    rmse = np.sqrt(((Y-obs)**2).mean())

    p_str=''
    if results.f_pvalue < 0.0001:
        p_str = 'p < 0.0001'
    elif results.f_pvalue < 0.001:
        p_str = 'p < 0.001'
    elif results.f_pvalue < 0.01:
        p_str = 'p < 0.01'
    elif results.f_pvalue < 0.05:
        p_str = 'p < 0.05'
    elif results.f_pvalue < 0.1:
        p_str = 'p < 0.1'
    else:
        p_str = 'p >= 0.1'
        
    ax.text(0.98,0.5,'y = %4.2fx + %4.2f\nR$^2$ = %4.2f; %s\nrmse = %4.1f %s; NSE = %4.2f' % (results.params[1],results.params[0],results.rsquared,p_str,rmse,units_str,nse),va='center',ha='right',transform=ax.transAxes,color='white')

    #x.set_aspect(1)
    #x.set_aspect('equal', adjustable='box-forced') 
    ax.set_xlim((0,mx))
    ax.set_ylim((0,mx))
    return 0



# Plot importances
# Inputs
# - ax: an axis object
# - varnames: the variable names (for annotation)
# - importances: the variable imporances
# - rank: (optional, default = True) rank variables by importance
# - c: (optional) colour (default taken from plasma colour map)
def plot_importances(ax,varnames,importances,rank=True,c=colour[0]):

    locs = np.arange(importances.size)+1
    imp = importances.copy()
    lab = np.asarray(varnames)
    if rank == True:
        imp = imp[np.argsort(importances)[::-1]]
        lab = lab[np.argsort(importances)[::-1]]

    ax.bar(locs,imp,color=c,linewidth=0,align='center')
    ax.set_ylabel('Variable importance',fontsize = axis_size)
    ax.set_xticks(locs)
    ax.set_xticklabels(lab,fontsize = axis_size)
    ax.set_xlim((0,locs.max()+1))
    return 0
