#==============================================================================
# calculate_FWI.py
#
# David T. Milodowski
# September 2017
#
# This module estimates fire risk based on the Canadian Fire Weather Index (FWI)
# danger rating system. This system combines meteorological conditions with an
# estimate of fuel moisture levels to determine an index of overall fire risk.
# For a detailed overview of the system, please see Stocks et al., The Forestry
# Chronicle, 1989
#
# The FWI is a "3-tiered" system comprising a series of indices that combine
# together to provide an integrated fire index system. It is more complex than
# the Australian McArthur system in the way it treats fuel moisture -
# separating out three different classes with different response times - but
# still lacks a treatment of fuel load abundance and diversity.
#
# The model components are summarised in the flowchart below
#
#        Level 1                       Level 2                   Level 3
#
# Fine Fuel Moisture Code ----> Initial Spread Index _
#         FFMC                          ISI           \
#                                                      \
#   Duff Moisture Code ___                              ---> Fire Weather Index
#         DMC              \                           /          FWI
#                           -----> Build-up Index ____/
#     Drought Code ________/            BUI
#          DC
#==============================================================================
import numpy as np

# Function to calculate the Fine Fuel Moisture Code.  This describes the
# moisture status of the fine litter, and is particularly important for in
# determining the ISI, as ignition likelihood is strongly dependent on the
# availability of fine litter.
# Inputs are:
# (1) relative humidity in %
# (2) air temperature in oC
def calculate_FFMC(H,T):

    # (1) Calculate rate constants for wetting and drying


    # (2) Calculate equilibrium moisture contents for fine fuel load. Units are
    #     in % moisture content based on dry weight.
    #     - equilibrium content for drying (Equation 8a)
    Ed = 0.942*H**0.679 + 11*np.exp((H-100)/10.) + 0.18*(21.1-T)*(1-np.exp(-0.115*H))
    #     - equilibrium content for wetting (Equation 8b)
    Ew = 0.618*H**0.753 + 10*np.exp((H-100)/10.) + 0.18*(21.1-T)*(1-np.exp(-0.115*H))
    
    # (3) Need to determine whether to apply wetting or drying phase.
    #     - If yesterday's moisture content, m0 is greater than Ed, drying regime
    #       prevails (equation 9)
    if m0>Ed:
        m = Ed + (m0 - Ed)*10**-kd
    #     - If m0 < Ew, wetting regime prevails (equation 10)
    elif m0<Ew:
        m = Ew - (Ew - m0)*10**-kw
    #     - otherwise no change in moisture
    else:
        m=m0
    


    return FFMC
