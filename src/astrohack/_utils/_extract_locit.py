import numpy as np

from casacore import tables as ctables
from astropy.coordinates import EarthLocation, AltAz, HADec, SkyCoord
from astropy.time import Time
import astropy.units as units
import xarray as xr
import time

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._tools import _casa_time_to_mjd, _altaz_to_hadec


def _extract_antenna_data(fname, ms_name):
    logger = _get_astrohack_logger()

    ant_table = ctables.table(ms_name + '::ANTENNA', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
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

    ant_pos_corrected = ant_pos + ant_off
    ant_rad = np.sqrt(ant_pos_corrected[:, 0] ** 2 + ant_pos_corrected[:, 1] ** 2 + ant_pos_corrected[:, 2] ** 2)
    ant_lat = np.arcsin(ant_pos_corrected[:, 2] / ant_rad)
    ant_lon = -np.arccos(ant_pos_corrected[:, 0] / (ant_rad * np.cos(ant_lat)))

    ant_dict = {'n_ant': n_ant, 'name': ant_nam, 'station': ant_sta, 'longitude': ant_lon, 'latitude': ant_lat,
                'radius': ant_rad, 'position': ant_pos, 'offset': ant_off, 'corrected': ant_pos_corrected}

    return ant_dict


def _extract_pointing_data(ms_name, ant_dict):
    pnt_dict = {}
    pnt_table = ctables.table(ms_name + '::POINTING', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    for i_ant in range(ant_dict['n_ant']):
        sub_table = ctables.taql("select DIRECTION, TIME, ANTENNA_ID from $pnt_table WHERE ANTENNA_ID == %s"
                                 % i_ant)
        # This is the pointing direction at pnt_time, in radians in AZ EL
        pnt_dir = sub_table.getcol('DIRECTION')
        az = pnt_dir[:, 0, 0]
        el = pnt_dir[:, 0, 1]
        pnt_time = sub_table.getcol('TIME')
        sub_table.close()

        ant_lat = ant_dict['latitude'][i_ant]
        ha, dec = _altaz_to_hadec(az, el, ant_lat)
        pnt_xds = xr.Dataset()
        coords = {"time": pnt_time}
        pnt_xds = pnt_xds.assign_coords(coords)
        pnt_xds["HOURANGLE"] = xr.DataArray(ha, dims="time")
        pnt_xds["DECLINATION"] = xr.DataArray(dec, dims="time")
        pnt_dict[ant_dict['name'][i_ant]] = pnt_xds
    pnt_table.close()

    return pnt_dict


def _extract_phase_gains(ms_name):
    main_table = ctables.table(ms_name, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    antenna1 = main_table.getcol('ANTENNA1')
    antenna2 = main_table.getcol('ANTENNA2')
    vis_time = main_table.getcol('TIME')
    data = main_table.getcol('DATA')[:, 0, 0]  # For The moment working with RR only, this should be a parameter
    corrected = main_table.getcol('CORRECTED_DATA')[:, 0, 0]

    orig_phase = np.angle(data)
    corr_phase = np.angle(corrected)
    phase_gains = corr_phase-orig_phase

    sel_gain, indices = np.unique(phase_gains, return_index=True)
    sel_time = vis_time[indices]
    sel_ant1 = antenna1[indices]
    sel_ant2 = antenna2[indices]
    time_order = np.argsort(sel_time)

    phase_dict = {'time': sel_time[time_order],
                 'antenna1': sel_ant1[time_order],
                 'antenna2': sel_ant2[time_order],
                 'baseline_phases': sel_gain[time_order]}

    return phase_dict


def _interpolate_pnt_times(pnt_dict, gain_dict):
    interp_pnt_dict = {}
    time_vis = gain_dict['time']
    for antenna in pnt_dict.keys():
        interp_pnt_dict[antenna] = pnt_dict[antenna].interp(time=time_vis, method="nearest")
    return interp_pnt_dict


def _derive_antenna_based_phases(gain_dict, ant_dict):
    # Build matrix with 1 and -1 to solve for phase gains!
    return
