#==============================================================================
# FWI_driver.py
#
# David T. Milodowski
# September 2017
#
# This is a simple driver function to test out the FWI module. It loads in the
# meteorological data, spins up the moisture codes and then iterates through
# the timesteps. At a later point, this will be incorporated into a dedicated
# object
#==============================================================================
# standard libraries
import numpy
from matplotlib import pyplot as plt

import sys
sys.paths.append('/exports/csce/datastore/geos/users/dmilodow/FOREST2020/EOdata/EO_data_processing/src/meteorology')

# own libraries
import calculate_FWI as FWI
import load_ERAinterim as era

#------------------------------------------------------------------------------
# locate files
path2files = '/disk/scratch/local.2/dmilodow/ERAinterim/source_files/0.175deg_Mexico'

# Set up fire risk simulation params
start_month = 1
start_year = 2000
end_month = 12
end_year = 2012

# Load in the met data
# - relative humidity in %
# - air temperature in oC
# - wind speed in m/s
# - pptn in mm
# - effective day length



