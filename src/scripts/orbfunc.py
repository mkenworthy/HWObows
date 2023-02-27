import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

import astropy.units as u

###########################################################################################

def phase(time, params):

        phases = ( ( (time - params['time periapsis passage']).jd ) /params['orbital period'].to(u.day).value)%1
        phases = np.array(phases).flatten()
        phases = convert_ma_to_ta(phases, params)
        
        return phases
        
###########################################################################################

def convert_ma_to_ta(ma, params):

        phases = np.arange(0,1,0.00001)
        f = interp1d(mean_anomaly(phases, params), phases, bounds_error=False, fill_value='extrapolate')

        return f(ma)

###########################################################################################

def mean_anomaly(true_anomaly, params):

        e = params['eccentricity']
        ta = true_anomaly*np.pi*2
        ma = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(ta/2)) + [2*np.pi if p>np.pi else 0 for p in ta] - e*(1-e**2)*np.sin(ta)/(1+e*np.cos(ta))

        return ma / (np.pi*2)

###########################################################################################

def orbit_radius(time, params):

        phases = phase(time, params)
        r = ( params['semi-major axis']*(1 - params['eccentricity']**2)/(1 + params['eccentricity']*np.cos(2 *np.pi*phases)) ).to(u.au)

        return np.array(r).flatten() * u.au

###########################################################################################

def coordinates3D(coords, time, params):

        aperi = -(params['argument of periapsis'] + params['periapsis precession']*(time - params['time periapsis passage']) ).to(u.rad).value
        inc = np.zeros(time.shape) + (np.pi/2 - params['inclination'].to(u.rad).value)
        lan = np.zeros(time.shape) - params['longditude of ascending node'].to(u.rad).value

        rotation_matrix = R.from_euler('ZXZ', np.array([lan, inc, aperi]).T)

        return rotation_matrix.apply(coords)

###########################################################################################

def xyz_position(time, params):

        phases = phase( time, params )
        r = orbit_radius( time, params )

        x = -r*np.cos(2 *np.pi*phases)
        y = r*np.sin(2 *np.pi*phases)

        xyz = np.array([x, y, np.zeros(len(r))]).T

        xyz = coordinates3D( xyz, time, params )

        return xyz * r.unit

##########################################################################################
