import numpy as np

import astropy.units as u
import astropy.coordinates as coord

def _calculate_parallactic_angle_chunk(time_samples, observing_location, direction, indicies, dir_frame='FK5', zenith_frame='FK5'):
    """
    Converts a direction and zenith (frame FK5) to a topocentric Altitude-Azimuth (https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html) 
    frame centered at the observing_location (frame ITRF) for a UTC time. The parallactic angles is calculated as the position angle of the Altitude-Azimuth 
    direction and zenith.
    
    Parameters
    ----------
    time_samples: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        UTC time series. Example '2019-10-03T19:00:00.000'.
    observing_location: int np.array, [3], [x,y,z], meters
        ITRF geocentric coordinates.
    direction: float np.array, [n_time,2], [time,ra,dec], radians
        The pointing direction.
    Returns
    -------
    parallactic_angles: float np.array, [n_time], radians
        An array of parallactic angles.
    """

    observing_location = coord.EarthLocation.from_geocentric(x=observing_location[0]*u.m, y=observing_location[1]*u.m, z=observing_location[2]*u.m)

    direction = np.take(direction, indicies, axis=0)

    direction = coord.SkyCoord(ra=direction[:,0]*u.rad, dec=direction[:,1]*u.rad, frame=dir_frame.lower())
    zenith = coord.SkyCoord(0, 90, unit=u.deg, frame=zenith_frame.lower())

    altaz_frame = coord.AltAz(location=observing_location, obstime=time_samples)
    zenith_altaz = zenith.transform_to(altaz_frame)
    direction_altaz = direction.transform_to(altaz_frame)

    return direction_altaz.position_angle(zenith_altaz).value