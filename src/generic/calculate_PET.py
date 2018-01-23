# calculate_PET.py
# Some functions to estimate potential evapotranspiration
import numpy as np
import calculate_day_length as cdl

# Initially just focus on the referencee PET indicated using the standard
# Penman-Monteith model, based on ERA-Interim meteorological reanalyses (or
# similar products).
# INputs:
# - maximum temperature at 2m in deg C
# - minimum temperature at 2m in deg C
# - wind speed at 2m in m s-1
# - total daily radiation in MJ m-2 d-1
# - surface air pressire in kPa
# - relative humidity in %
# - date in np.datetim64 format
# - latitude
# - elevation above sea level in m (optional)

def calculate_PET_penman_monteith(mxT2m,mnT2m, u2m, Rs, P, RH, date, latitude, z=np.array([])):

    # estimate average temperature at 2m
    Tmean = (mxT2m+mnT2m)/2.

    # get slope of saturation vapour pressure curve
    delta = 4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3)))/(Tmean+237.3)**2

    # Psychrometric constant
    psy = 0.000665*P

    # Delta term
    DT = delta/(delta+psy*(1+0.34*u2m))

    # Psi term
    PT = psy/(delta+psy*(1+0.34*u2m))

    # Temperature term
    TT = (900/(Tmean+273))*u2m

    # Mean saturation vapour pressure derived from air temperature
    esTmax = 0.6108*np.exp(17.27*mxT2m/(mxT2m+273.3))
    esTmin = 0.6108*np.exp(17.27*mnT2m/(mnT2m+273.3))
    es = (esTmax + esTmin)/2.

    # Actual vapour pressure derived from relative humidity
    ea = es*RH/200.

    # Inverse relative distance to Sun
    DoY = cdl.calculate_DoY(date)
    dr = 1 + 0.033*np.cos(2*np.pi/365.*DoY)

    # declination of the sun
    #dec = cdl.calculate_declination(date)
    dec = 0.409*np.sin(2*np.pi/365.*DoY-1.39)
    
    # latitude in radians
    lat_rad = np.pi/180.*latitude

    #sunset hour angle
    sunset_HA = np.arccos(-np.tan(lat_rad)*np.tan(dec))

    # ET radiation, Ra, in MJ m-2 d-1
    Gsc = 0.0820 # MJ m-2 min-1
    Ra = 24.*60./np.pi*Gsc*dr*(sunset_HA*np.sin(lat_rad)*np.sin(dec)+np.cos(lat_rad)*np.cos(dec)*np.sin(sunset_HA))
    
    # Clear sky solar radiation
    if z.size>0:
        Rso = (0.75+2*10**-5*z)*Ra
    else:
        Rso = 0.75*Ra
    
    # Net shortwave radiation
    a = 0.23 # albedo for reference crop
    Rns = (1-a)*Rs

    # Net outgoing longwave radiation
    boltz = 4.903*10**-9 # MJ K-1 m-2 d-1
    Rnl = boltz*((mxT2m+273.16)**4+(mnT2m+273.16)**4)/2.*(0.34-0.14*np.sqrt(ea))*(1.35*Rs/Rso - 0.35)

    # Net radiation, Rn
    Rn = Rns - Rnl
    # expressed in terms of evaporation equivalent (mm)
    Rng = 0.408*Rn

    # Overall ET
    ETrad = DT*Rng
    ETwind = PT*TT*(es-ea)
    PET= ET_rad + ETwind

    return PET

# Calculate climatic  water deficit, cwd
# CWD defined as the cumulative deficit between precipitation and PET
# Time dimension assumed to be axis 0
# spin_up_time is the initial period used to spin up the cwd
def calculate_cwd(precipitation,PET,spin_up_time = 365):
    n_tsteps = precipitation.shape[0]
    cwd = np.zeros(precipitation.shape)
    for tt in range(0,spin_up_time):
        cwd[0] = np.min(0,cwd[0]+precipitation[tt]-PET[tt])

    cwd[0] = np.min(0,cwd[0]+precipitation[0]-PET[0])
    for tt in range(1,n_tsteps):
        cwd[tt] = np.min(0,cwd[tt-1]+precipitation[tt]-PET[tt])

    return cwd
