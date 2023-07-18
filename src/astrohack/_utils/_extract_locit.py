import numpy as np

from casacore import tables as ctables
from astropy.coordinates import EarthLocation, AltAz, HADec, SkyCoord
from astropy.time import Time
import astropy.units as units
import xarray as xr
import time

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._tools import _casa_time_to_mjd, _altaz2hadec


def _extract_antenna_data(fname, ms_name):

    logger = _get_astrohack_logger()

    ant_table = ctables.table(ms_name+'::ANTENNA', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    ant_off = ant_table.getcol('OFFSET')
    ant_pos = ant_table.getcol('POSITION')
    ant_mnt = ant_table.getcol('MOUNT')
    ant_nam = ant_table.getcol('NAME')
    ant_sta = ant_table.getcol('STATION')
    ant_typ = ant_table.getcol('TYPE')
    ant_table.close()

    n_ant = ant_off.shape[0]

    error = False
    for iant in range(n_ant):
        if ant_mnt[iant] != 'ALT-AZ':
            logger.error(f'[{fname}]: Antenna {ant_nam[iant]} has a non supported mount type: {ant_mnt[iant]}')
            error = True
        if ant_typ[iant] != 'GROUND-BASED':
            error = True
            logger.error(f'[{fname}]: Antenna {ant_nam[iant]} is not ground based which is currently not supported')

    if error:
        msg = f'[{fname}]: Unsupported antenna types'
        logger.error(msg)
        raise Exception(msg)

    ant_pos_corrected = ant_pos+ant_off
    ant_rad = np.sqrt(ant_pos_corrected[:, 0]**2 + ant_pos_corrected[:, 1]**2 + ant_pos_corrected[:, 2]**2)
    ant_lat = np.arcsin(ant_pos_corrected[:, 2]/ant_rad)
    ant_lon = -np.arccos(ant_pos_corrected[:, 0] / (ant_rad*np.cos(ant_lat)))

    # for i in range(ant_off.shape[0]):
    #     print(ant_nam[i], ant_lon[i]*180/np.pi, ant_lat[i]*180/np.pi)
    ant_dict = {'n_ant': n_ant, 'name': ant_nam, 'station': ant_sta, 'longitude': ant_lon, 'latitude': ant_lat,
                'radius': ant_rad, 'position': ant_pos, 'offset': ant_off, 'corrected': ant_pos_corrected}

    return ant_dict


def _extract_pointing_data(fname, ms_name, ant_dict):

    az_el_pnt_dict = {}
    pnt_table = ctables.table(ms_name + '::POINTING', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    for i_ant in range(ant_dict['n_ant']):
        sub_table = ctables.taql("select DIRECTION, TIME, ANTENNA_ID from $pnt_table WHERE ANTENNA_ID == %s"
        % (i_ant))
        #sub_table = ctables.taql(f"select ANTENNA_ID, DIRECTION, TIME, from $pnt_table WHERE ANTENNA_ID == {i_ant}")
        # This is the pointing direction at pnt_time, in radians in AZ EL
        pnt_dir = sub_table.getcol('DIRECTION')
        az = pnt_dir[:, 0, 0]
        el = pnt_dir[:, 0, 1]
        pnt_time = sub_table.getcol('TIME')
        sub_table.close()

        x_ant, y_ant, z_ant = ant_dict['position'][i_ant]
        print(ant_dict['name'][i_ant], x_ant, y_ant, z_ant)
        ant_lat = ant_dict['latitude'][i_ant]

        t0 = time.time()
        ant_pos = EarthLocation.from_geocentric(x_ant, y_ant, z_ant, 'meter')
        mjd_time = Time(_casa_time_to_mjd(pnt_time), format='mjd', scale='utc')
        az_el_frame = AltAz(location=ant_pos, obstime=mjd_time)
        ha_dec_frame = HADec(location=ant_pos, obstime=mjd_time)
        azel_coor = SkyCoord(az*units.rad, el*units.rad, frame=az_el_frame)
        ha_dec_coor = azel_coor.transform_to(ha_dec_frame)
        t1 = time.time()
        ha, dec = _altaz2hadec(az, el, ant_lat)
        t2 = time.time()
        print('astropy:', t1-t0)
        print('daniel:', t2-t1)
        #az_el_pnt_dict[ant_dict['name'][i_ant]] = sub_dict

    pnt_table.close()

    return az_el_pnt_dict


def _extract_phase_gains(ms_name, pol_state, phase_gain_tol):
    time_vis = None
    phase_gains = None
    return time_vis, phase_gains
