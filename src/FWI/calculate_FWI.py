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
# (3) wind speed in m/s
# (4) Precipitation in mm
# (5) Previous days FFMC value (default is -9999, for which we need to estimate
#     starting FFMC)
def calculate_FFMC(H,T,W,P,FFMC0=-9999):

    if FFMC0==-9999:
        FFMC0 = 60. # currently supply a default value, but need to develop 
                    # scheme based on local climate
    m0 = 147.2*(101-FFMC0)/(59.5+FFMC0)
    
    # (1) Calculate rate constants for wetting and drying
    #     Wetting and drying rates assumed to be exponential, with rate 
    #     constants a function of temperature, windspeed and relative humidity
    #     - drying
    k0_d = 0.424*(1-(H/100)**1.7) + 0.0694*W**0.5*(1-(H/100)**8) # equation 4
    kd = k0_d * 0.581*np.exp(0.00365*T) # equation 6
    #     - wetting
    k0_w = 0.424*(1-((100-H)/100)**1.7) + 0.0694*W**0.5*(1-((100-H)/100)**8) # equation 5
    kd = k0_d * 0.581*np.exp(0.00365*T) # equation 6

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
        m=m0.copy()
    
    # (4) Now deal with moisture input due to rain
    #     This decreases with increasing pptn rate and decreases with increased 
    #     moisture content
    #     Assume intercepted rainfall accounts for 0.5 mm
    interecepted_mm = 0.5
    if P>intercepted_mm:
        P_ = P-intercepted_mm
        m += P_*42.5*np.exp(-100/(251-m0))*(1-np.exp(-6.93/P_)) # equation 12
        if m0>150:
            m+=0.0015*(m0-150)**2*P_**0.5 # equation 13
    
    FFMC = 59.5*(250-m)/(147.2+m)
    return FFMC

# Function to calculate the Duff Moisture Code (DMC). This deals with the
# moisture content of the duff layer - the upper layers of the forest floor,
# which are beginning to decay. Moisture in this layer is gained through inputs
# from rainfall, and lost through an exponential drying function towards an
# assumed equilibrium moisture content of 20%.
# Inputs are:
# (1) relative humidity in %
# (2) air temperature in oC
# (3) Precipitation in mm
# (4) Effective day length in hours, Le, assumed to be 3hrs less than actual day
#     length
# (5) Previous days DMC value (default is -9999, for which we need to estimate
#     starting DMC)
def calculate_DMC(H,T,P,Le,DMC0=-9999):

    if DMC0==-9999:
        DMC0 = 20. # currently supply a default value, but need to develop
                   # scheme based on local climate
    m0 = 20+np.exp((DMC0-244.73)/(-43.43)) # equation 16 rearranged

    # (1) Calculate rate constant for drying of duff layer, kd
    #     - Assumes exponential drying, equilibrium moisture content 20%
    #     - Assumes kd proportional to temperature and relative "dryness"
    #     - Assumes kd proportional to day length -3 hours
    kd = 1.894*(T+1.1)*(100-H)*Le*10**-6  # no equation number in manuscript

    # (2) Calculate wetting of DMC due to rain. Assumes increase in moisture
    #     inversely proportional to intensity of rainfall, and that wetting
    #     effect also decreases as initial moisture content increases
    precipitation_threshold_mm = 1.5
    m=m0.copy()
    DMC = DMC0.copy()
    if P>precipitation_threshold_mm:
        P_=0.92*P-1.27 # equation 17
        b = 0
        # equations 19 a,b,c
        if DMC0 <= 33:
            b = 100/(0.5+0.3*DMC0)
        elif DMC0 <= 65:
            b = 14 - 1.3*np.log(DMC0)
        else:
            b = 6.2*np.log(DMC0) - 17.2

        m+=1000*P_/(48.77+b*P_) # equation 18
        
        # Now recalculate DMC
        DMC = 244.72 - 43.43*np.log(m-20) # equation 16

    # (3) Now account for drying 
    DMC += 100*kd # no equation number in manuscript
    
    return DMC
