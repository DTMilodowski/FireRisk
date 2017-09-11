#==============================================================================
# calculate_Mark5.py
#
# David T. Milodowski
# September 2017
#
# This module estimates fire risk based on the Australian McArthur (Mark5) fire
# danger rating system. This is a relatively simple fire risk metric
# predominately determined by meteorological conditions. The original equations
# were published by Noble et al., Aust. J. Ecol., 1980.
#
# Note that the Mark5 index does not account for factors such as diversity in 
# the abundance or characteristics of fuel loads, or topographic slope, and is
# based on an empirical formulation of fire risk from Australia. It should
# therefore be considered as a tool to understand the weather/climate influence
# on fire risk, rather than as a comprehensive metric of fire risk.
#
# Meteorological variables required:
# - relative humidity, H
# - temperature, T
# - wind velocity, U
# - precipitation, P
# - number of days since previous rainfall event, d
# - Keetch-Byram drought index [Keetch & Byram, 1960], I
#==============================================================================
import numpy as np

def calculate_Keetch_Byram_drought_index():

def calculate_Mark5(H,T,U,P,d,I):

    # First calculate drought factor
    DF = (0.191*(I+104)*(d+1)**1.5)/(3.52*(d+1)**1.5 + P-1)

    # Now use this to calculate fire risk
    F = 2.0*exp(-0.450 + 0.987*np.log(DF) - 0.0345*H + 0.0338*T + 0.0234*U)

