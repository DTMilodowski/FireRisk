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
# - Keetch-Byram drought index [Keetch & Byram, 1968], KB
#
# By including the K-B drought index, the Mark5 approach does attempt to
# capture some of the dynamics of soil and litter moisture on fire risk.
#==============================================================================
import numpy as np

# Calculation of K-B drought index for a given day, requires:
# (1) Moisture deficiency at start of the day, Q0 in mm
# (2) max temperature in oC
# (3) total rainfall for past 24 hours in mm
# (4) MAP in mm. 
# This was originally a book-keeping approach based on a lookup table, with the # index updated at each iteration by removing net precipitation and incrementing
# the index to account for drying.
def calculate_Keetch_Byram_drought_index(Q0, maxT, P24, MAP, dt_days = 1, interception_mm = 5):

    # (1) account for moisture addition due to precipitation
    dQ_pptn = -np.max(0,P24-interception_mm)
    
    # (2) account for drying due to evapotranspiration
    dQ_drying = (10**-3)*(203.2-Q0)*(0.968*np.exp(0.0875*maxT + 1.5552) -8.30)*dt_days/(1+ 10.88*np.exp(-0.001736*MAP))
    
    # (3) update moisture deficiency
    Q = Q0 + dQ_pttn + dQ_drying

    return Q

# Calculation of the Mark5 index based on weather and K-B drought index
def calculate_Mark5(H,T,U,P,d,I):

    # First calculate drought factor
    DF = (0.191*(KB+104)*(d+1)**1.5)/(3.52*(d+1)**1.5 + P-1)

    # Now use this to calculate fire risk
    F = 2.0*exp(-0.450 + 0.987*np.log(DF) - 0.0345*H + 0.0338*T + 0.0234*U)

