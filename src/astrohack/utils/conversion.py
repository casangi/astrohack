import toolviper.utils.logger as logger
from astropy import units as units
from astropy.coordinates import EarthLocation, AltAz, HADec, SkyCoord
from astropy.time import Time

from astrohack.utils.constants import *


# Global conversion functions
def to_db(val: float):
    """
    Converts a float value to decibels
    Args:
        val (float): Value to be converted to decibels
    Returns:
        Value in decibels
    """
    return 10.0 * np.log10(val)


def convert_unit(unitin, unitout, kind):
    """
    Convert between unit of the same kind
    Args:
        unitin: Origin unit
        unitout: Destination unit
        kind: 'trigonometric' or 'length'

    Returns:
        Conversion factor to go from unitin to unitout
    """
    try:
        unitlist = unit_dict[kind]
        factorlist = fact_dict[kind]

    except KeyError:

        logger.error("Unrecognized unit kind: " + kind)
        raise KeyError('Unrecognized unit kind')

    inidx = _test_unit(unitin, unitlist)
    ouidx = _test_unit(unitout, unitlist)
    factor = factorlist[inidx] / factorlist[ouidx]

    return factor


def _test_unit(unit, unitlist):
    """
    Test if a unit is known
    Args:
        unit: unit name
        unitlist: List containing unit names

    Returns:
        Unit index in unitlist
    """
    try:
        idx = unitlist.index(unit)
    except ValueError:
        logger.error("Unrecognized unit: " + unit)
        raise ValueError('Unit not in list')

    return idx


def to_stokes(grid, pol):
    grid_stokes = np.zeros_like(grid)

    if 'RR' in pol:
        grid_stokes[:, :, 0, :, :] = (grid[:, :, 0, :, :] + grid[:, :, 3, :, :]) / 2
        grid_stokes[:, :, 1, :, :] = (grid[:, :, 1, :, :] + grid[:, :, 2, :, :]) / 2
        grid_stokes[:, :, 2, :, :] = 1j * (grid[:, :, 1, :, :] - grid[:, :, 2, :, :]) / 2
        grid_stokes[:, :, 3, :, :] = (grid[:, :, 0, :, :] - grid[:, :, 3, :, :]) / 2
    elif 'XX' in pol:
        grid_stokes[:, :, 0, :, :] = (grid[:, :, 0, :, :] + grid[:, :, 3, :, :]) / 2
        grid_stokes[:, :, 1, :, :] = (grid[:, :, 0, :, :] - grid[:, :, 3, :, :]) / 2
        grid_stokes[:, :, 2, :, :] = (grid[:, :, 1, :, :] + grid[:, :, 2, :, :]) / 2
        grid_stokes[:, :, 3, :, :] = 1j * (grid[:, :, 1, :, :] - grid[:, :, 2, :, :]) / 2
    else:
        raise Exception("Pol not supported " + str(pol))

    return grid_stokes


def convert_dict_from_numba(func):
    def wrapper(*args, **kwargs):
        numba_dict = func(*args, **kwargs)

        converted_dict = dict(numba_dict)

        for key, _ in numba_dict.items():
            converted_dict[key] = dict(converted_dict[key])

        return converted_dict

    return wrapper


def altaz_to_hadec_astropy(az, el, time, x_ant, y_ant, z_ant):
    """
    Astropy conversion from Alt Az to Ha Dec, seems to be more precise, but it is VERY slow
    Args:
        az: Azimuth
        el: Elevation
        time: Time
        x_ant: Antenna x position in geocentric coordinates
        y_ant: Antenna y position in geocentric coordinates
        z_ant: Antenna z position in geocentric coordinates

    Returns: Hour angle and Declination

    """
    ant_pos = EarthLocation.from_geocentric(x_ant, y_ant, z_ant, 'meter')
    mjd_time = Time(casa_time_to_mjd(time), format='mjd', scale='utc')
    az_el_frame = AltAz(location=ant_pos, obstime=mjd_time)
    ha_dec_frame = HADec(location=ant_pos, obstime=mjd_time)
    azel_coor = SkyCoord(az * units.rad, el * units.rad, frame=az_el_frame)
    ha_dec_coor = azel_coor.transform_to(ha_dec_frame)

    return ha_dec_coor.ha, ha_dec_coor.dec


def hadec_to_elevation(hadec, lat):
    """Convert HA + DEC to elevation.

    (HA [rad], dec [rad])

    Provided by D. Faes DSOC
    """
    #
    cosha = np.cos(hadec[0])
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    sin_el = sinlat * np.sin(hadec[1]) + coslat * np.cos(hadec[1]) * cosha
    el = np.arcsin(sin_el)
    return el


def hadec_to_altaz(ha, dec, lat):
    """Convert HA + DEC to Alt + Az coordinates.

    (HA [rad], dec [rad])

    Provided by D. Faes DSOC
    """
    #
    sinha = np.sin(ha)
    cosha = np.cos(ha)
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    bottom = cosha * sinlat - np.tan(dec) * coslat
    sin_el = sinlat * np.sin(dec) + coslat * np.cos(dec) * cosha
    az = np.arctan2(sinha, bottom)
    el = np.arcsin(sin_el)
    az += np.pi  # formula is starting from *South* instead of North
    if az > 2 * np.pi:
        az -= 2 * np.pi
    return az, el


def altaz_to_hadec(az, el, lat):
    """Convert AltAz to HA + DEC coordinates.

    (HA [rad], dec [rad])

    Provided by D. Faes DSOC
    """
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinel = np.sin(el)
    cosel = np.cos(el)
    cosaz = np.cos(az)
    sindec = sinlat * sinel + coslat * cosel * cosaz
    dec = np.arcsin(sindec)
    argarccos = (sinel - sinlat * sindec) / (coslat * np.cos(dec))
    lt1 = argarccos < -1
    argarccos[lt1] = -1.0
    ha = np.arccos(argarccos)
    return ha, dec


def casa_time_to_mjd(times):
    corrected = times / 3600 / 24.0
    return corrected
