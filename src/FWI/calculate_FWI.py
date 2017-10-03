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
    kw = k0_w * 0.581*np.exp(0.00365*T) # equation 6

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

# Complimentary function to deal with numpy arrays efficiently. In this case the
# inital conditions are a required input argument
def calculate_FFMC_array(H,T,W,P,FFMC0=-9999):

    m0 = 147.2*(101-FFMC0)/(59.5+FFMC0)
    m = m0.copy()
    
    # (1) Calculate rate constants for wetting and drying
    #     Wetting and drying rates assumed to be exponential, with rate 
    #     constants a function of temperature, windspeed and relative humidity
    #     - drying
    k0_d = 0.424*(1.-(H/100.)**1.7) + 0.0694*W**0.5*(1.-(H/100.)**8) # equation 4
    kd = k0_d * 0.581*np.exp(0.00365*T) # equation 6
    #     - wetting
    k0_w = 0.424*(1.-((100-H)/100.)**1.7) + 0.0694*W**0.5*(1.-((100-H)/100.)**8) # equation 5
    kw = k0_w * 0.581*np.exp(0.00365*T) # equation 6

    # (2) Calculate equilibrium moisture contents for fine fuel load. Units are
    #     in % moisture content based on dry weight.
    #     - equilibrium content for drying (Equation 8a)
    Ed = 0.942*H**0.679 + 11*np.exp((H-100)/10.) + 0.18*(21.1-T)*(1-np.exp(-0.115*H))
    #     - equilibrium content for wetting (Equation 8b)
    Ew = 0.618*H**0.753 + 10*np.exp((H-100)/10.) + 0.18*(21.1-T)*(1-np.exp(-0.115*H))
    
    # (3) Need to determine whether to apply wetting or drying phase.
    #     - If yesterday's moisture content, m0 is greater than Ed, drying regime
    #       prevails (equation 9)
    dry_mask = m0>Ed
    m[dry_mask] = Ed[dry_mask] + (m0[dry_mask] - Ed[dry_mask])*10**-kd[dry_mask]
    #     - If m0 < Ew, wetting regime prevails (equation 10)
    wet_mask = m0<Ew
    m[wet_mask] = Ew[wet_mask] - (Ew[wet_mask] - m0[wet_mask])*10**-kw[wet_mask]
    #     - otherwise no change in moisture
    
    # (4) Now deal with moisture input due to rain
    #     This decreases with increasing pptn rate and decreases with increased 
    #     moisture content
    #     Assume intercepted rainfall accounts for 0.5 mm
    intercepted_mm = 0.5
    P_ = P-intercepted_mm

    # mask out cells for which there is insufficient rainfall to impact fine 
    # litter layer
    P_mask = P>intercepted_mm

    m[P_mask] += P_[P_mask]*42.5*np.exp(-100/(251-m0[P_mask]))*(1-np.exp(-6.93/P_[P_mask])) # equation 12
    # additional correction factor to account for case where moisture content is
    # particularly high
    m0_mask = m0[P_mask]>150
    m[P_mask][m0_mask]+=0.0015*(m0[P_mask][m0_mask]-150)**2*P_[P_mask][m0_mask]**0.5 # equation 13
    
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

# complimentary function to process arrays in calculation of DMC. Note that now
# the initial DMC conditions need to be ingested as an argument, therefore need
# predefining in advance.
def calculate_DMC_array(H,T,P,Le,DMC0):

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
    b = np.zeros(P.shape)
    
    # generate a mask to highlight cells that have received sufficient rainfall
    # to impact on duff layer
    P_mask = P>precipitation_threshold_mm

    P_=0.92*P-1.27 # equation 17

    # equations 19 a,b,c
    DMC_mask_a = DMC0[P_mask] <= 33
    DMC_mask_b = np.all((DMC0[P_mask] <= 33, DMC0[P_mask] <= 65),axis=0)
    DMC_mask_c = DMC0[P_mask] > 65
    b[P_mask][DMC_mask_a] = 100/(0.5+0.3*DMC0[P_mask][DMC_mask_a])
    b[P_mask][DMC_mask_b] = 14 - 1.3*np.log(DMC0[P_mask][DMC_mask_b])
    b[P_mask][DMC_mask_c] = 6.2*np.log(DMC0[P_mask][DMC_mask_c]) - 17.2

    m[P_mask]+=1000*P_[P_mask]/(48.77+b[P_mask]*P_[P_mask]) # equation 18
        
    # Now recalculate DMC
    DMC[P_mask] = 244.72 - 43.43*np.log(m[P_mask]-20) # equation 16

    # (3) Now account for drying 
    DMC += 100*kd # no equation number in manuscript
    return DMC

# Function to calculate the Drought Code, DC, which describes the moisture 
# content of the soil organic carbon, and has the slowest response time of the
# carbon fuels.  It is important however in determining the extent to which a
# fire may develop once ignition has occurred, and therefore is integrated into
# the Build Up Index.
# -Assumes a maximum theoretical moisture content of 800 (what are the units???)
# -The original formulation makes an adjustment to account for "overwintering"
#  i.e. snow melt effects. As snowfall is negligible across most of the tropics
#  I ignore this.
# -The original formulation also uses an adjustment factor, Lf, to account for
#  seasonal changes in day length. I ignore this also, based on the fact that
#  a value estimated for Canada is unlikely to be valid in the tropics, while
#  in any case, day length is relatively insensitive to season due to the low
#  lattitudes
# Inputs are:
# (1) air temperature in oC
# (2) Precipitation in mm
# (3) Previous days DC value (default is -9999, for which we need to estimate
#     starting DC)
def calculate_DC(T,P, DC0 = -9999, Lf = 0):
    if DC0==-9999:
        DC0 = 200. # currently supply a default value, but need to develop
                   # scheme based on local climate
    m0 = np.exp(-DC0/400.)*800. # equation 22

    # (1) rainfall phase. Assume that rainfall must be greater than 2.8 mm to
    #     impact on soil moisture (I appreciate this is arbitrary!)
    m = m0.copy()
    precipitation_threshold_mm = 2.8
    if P>precipitation_threshold_mm:
        P_=0.83*P-1.27 # equation 23
        m+=3.937*P_    # equation 24

    # (2) drying phase, i.e. potential evapotranspiration, V.  Use an empirical
    #     equation (no idea what data this is based on).
    V = 0.36*(T+2.8) + Lf

    # (3) calculate DC based on equation 22 and 26
    DC = 400*np.log(800./m) + 0.5*V
    return DC

# complimentary function to process arrays rather than individual floats
# Note that initial DC required in this case (no default value), so this will
# need pre-calculating
def calculate_DC_array(T,P, DC0, Lf = 0):
    m0 = np.exp(-DC0/400.)*800. # equation 22

    # (1) rainfall phase. Assume that rainfall must be greater than 2.8 mm to
    #     impact on soil moisture (I appreciate this is arbitrary!)
    m = m0.copy()
    precipitation_threshold_mm = 2.8
    mask = P>precipitation_threshold_mm
    P_=0.83*P-1.27             # equation 23
    m[mask]+=3.937*P_[mask]    # equation 24

    # (2) drying phase, i.e. potential evapotranspiration, V.  Use an empirical
    #     equation (no idea what data this is based on).
    V = 0.36*(T+2.8) + Lf

    # (3) calculate DC based on equation 22 and 26
    DC = 400*np.log(800./m) + 0.5*V
    return DC

# Function to calculate the Initial Spread Index (ISI).
# The ISI represents the risk of a fire igniting and spreading to become
# significant. It is determined by the FFMC and the wind speed. Topography is
# not taken into account (unlike the USFS NDFRS).
# The relationship between fire spread and windspeed is assumed to follow an
# exponential relationship.
# Inputs are:
# (1) FFMC
# (2) Wind speed measured at 10 m in km/h
def calculate_ISI(FFMC,W):
    
    # (1) calculate wind speed effect
    fW = np.exp(0.05039*W)
    
    # (2) calculate fuel moisture effect
    m = 147.2*(101-FFMC)/(59.5+FFMC)
    fm = 91.9*np.exp(-0.1386*m)*(1+(m**5.31)/(4.93*10**7))

    # (3) combine into ISI
    ISI = 0.208*fW*fm
    return ISI

# Creating complementary set of functions for clarity - in this case, identical
# to the above
def calculate_ISI_array(FFMC,W):
    
    # (1) calculate wind speed effect
    fW = np.exp(0.05039*W)
    
    # (2) calculate fuel moisture effect
    m = 147.2*(101-FFMC)/(59.5+FFMC)
    fm = 91.9*np.exp(-0.1386*m)*(1+(m**5.31)/(4.93*10**7))

    # (3) combine into ISI
    ISI = 0.208*fW*fm
    return ISI

# Function to calculate the Build-up Index (BUI)
# Combines the duff and soil organic matter moisture levels to give an 
# integrated picture of the potential fuel available for surface fuel
# consumption, which would be required to maintain a fire spread.  It does not
# account for actual fuel loads, however, and therfore assumes that fires are
# not limited by fuel availability.  DC is downweighted relative to DMC to
# reflect the relative importances in governing overall fuel loads. DC will only
# contribute when DMC levels are significantly greater than zero.
# Note that as DC responds more slowly than DMC, the BUI will exhibit seasonal
# hysteresis.
# Inputs:
# (1) DMC
# (2) DC
def calculate_BUI(DMC,DC):
    BUI = 0.8*DMC*DC/(DMC+0.4*DC) # equation 36
    return BUI

# Creating complementary set of functions for clarity - in this case, identical
# to the above
def calculate_BUI_array(DMC,DC):
    BUI = 0.8*DMC*DC/(DMC+0.4*DC) # equation 36
    return BUI

# Function to calculate the Forest Weather Index (FWI)
# This is the overarching fire risk index combining the ISI - the risk of fire
# initiation and spread - with the BUI - the availabile fuel load to sustain a
# significant fire. It is analagous to Byram's fireline intensity
# Inputs:
# (1) ISI
# (2) BUI
def calculate_FWI(ISI,BUI):

    # (1) calculate the duff moisture function, fBUI, which attempts to 
    #     translate the available fuel (BUI) into an index of burn intensity
    #     (equations 38 a and b)
    fBUI=0
    if BUI<=80.:
        fBUI = 0.626*BUI**0.809 +2
    else:
        fBUI = 1000./(25. + 108.64*np.exp(-0.023*BUI))

    # (2) Now calculate the FWI
    B = 0.1 * ISI * fBUI
    FWI = B.copy()
    if B>=1:
        FWI = np.exp(2.72*(0.434*np.log(B))**0.647)
    return FWI

# equivalent function reading in numpy arrays rather than individual float
# values 
def calculate_FWI_array(ISI,BUI):

    # (1) calculate the duff moisture function, fBUI, which attempts to 
    #     translate the available fuel (BUI) into an index of burn intensity
    #     (equations 38 a and b)
    fBUI=np.zeros(ISI.shape)
    fBUI[BUI<=80] = 0.626*BUI[BUI<=80]**0.809 + 2.
    fBUI[BUI>80] = 1000./(25. + 108.64*np.exp(-0.023*BUI[BUI>80]))

    # (2) Now calculate the FWI
    B = 0.1 * ISI * fBUI
    FWI = B.copy()
    FWI[B>=1] = np.exp(2.72*(0.434*np.log(B[B>=1]))**0.647)
    return FWI
