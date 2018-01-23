# calculate_day_length.py
# a python implementation of the NOAA day length calculator
# https://www.esrl.noaa.gov/gmd/grad/solcalc/
import numpy as np

def calculate_day_length(julian_day,latitude):
    julian_century = (julian_day-2451545.)/36525.
    geom_mean_long_sun = 280.46646+julian_century*(36000.76983 + julian_century*0.0003032) % 360.
    geom_mean_anom_sun = 357.52911+julian_century*(35999.05029 - 0.0001537*julian_century)
    eccentricity_earth_orbit = 0.016708634-julian_century*(0.000042037+0.0000001267*julian_century)
    
    sun_eq_of_ctr = np.sin(geom_mean_anom_sun*np.pi/180.)*(1.914602-julian_century*(0.004817+0.000014*julian_century))+np.sin(np.pi*2*geom_mean_anom_sun/180.)*(0.019993-0.000101*julian_century)+np.sin(np.pi*3*geom_mean_anom_sun/180.)*0.000289
    sun_true_long = geom_mean_long_sun+sun_eq_of_ctr
    sun_true_anom = geom_mean_anom_sun+sun_eq_of_ctr
    
    sun_rad_vector = (1.000001018*(1-eccentricity_earth_orbit**2))/(1+eccentricity_earth_orbit*np.cos(np.pi*sun_true_anom/180.))
    sun_app_long = sun_true_long-0.00569-0.00478*np.sin(np.pi*(125.04-1934.136*julian_century)/180.)

    mean_obliq_eliptic = 23+(26+((21.448-julian_century*(46.815+julian_century*(0.00059-julian_century*0.001813))))/60.)/60.
    obliq_corr = mean_obliq_eliptic+0.00256*np.cos(np.pi*(125.04-1934.136*julian_century)/180.)

    sun_rt_ascen = (180./np.pi)*(np.arctan2(np.cos((np.pi/180.)*sun_app_long),np.cos((np.pi/180.)*obliq_corr)*np.sin(np.pi/180.*sun_app_long)))
    
    sun_declin = (180/np.pi)*(np.arcsin(np.sin((np.pi/180.)*obliq_corr)*np.sin((np.pi/180.)*sun_app_long)))

    var_y = np.tan((np.pi/180.)*(obliq_corr/2.))*np.tan((np.pi/180.)*(obliq_corr/2.))

    eq_of_time = 4*(180/np.pi)*(var_y*np.sin(2*(np.pi/180.)*geom_mean_long_sun)-2*eccentricity_earth_orbit*np.sin((np.pi/180.)*geom_mean_anom_sun)+4*eccentricity_earth_orbit*var_y*np.sin((np.pi/180.)*geom_mean_anom_sun)*np.cos(2*(np.pi/180.)*geom_mean_long_sun)-0.5*var_y**2*np.sin(4*(np.pi/180.)*geom_mean_long_sun)-1.25*eccentricity_earth_orbit**2*np.sin(2*(np.pi/180.)*(geom_mean_anom_sun)))

    HA_sunrise = (180/np.pi)*(np.arccos(np.cos((np.pi/180.)*90.833)/(np.cos((np.pi/180.)*latitude)*np.cos((np.pi/180.)*sun_declin))-np.tan((np.pi/180.)*latitude)*np.tan((np.pi/180.)*sun_declin)))
    
    sunlight_duration = 8*HA_sunrise

    print sunlight_duration
    
    sunlight_duration_hrs = sunlight_duration/60.

    return sunlight_duration_hrs
