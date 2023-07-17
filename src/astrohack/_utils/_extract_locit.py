from casacore import tables as ctables
import numpy as np
import xarray as xr

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger


def _extract_antenna_data(fname, ms_name):

    logger = _get_astrohack_logger()

    anttab = ctables.table(ms_name+'::ANTENNA', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    ant_off = anttab.getcol('OFFSET')
    ant_pos = anttab.getcol('POSITION')
    ant_mnt = anttab.getcol('MOUNT')
    ant_nam = anttab.getcol('NAME')
    ant_sta = anttab.getcol('STATION')
    ant_typ = anttab.getcol('TYPE')

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

    ant_pos_cart = ant_pos+ant_off
    ant_rad = np.sqrt(ant_pos_cart[:, 0]**2 + ant_pos_cart[:, 1]**2 + ant_pos_cart[:, 2]**2)
    ant_lat = np.arcsin(ant_pos_cart[:, 2]/ant_rad)
    ant_lon = -np.arccos(ant_pos_cart[:, 0] / (ant_rad*np.cos(ant_lat)))

    for i in range(ant_off.shape[0]):
        print(ant_nam[i], ant_lon[i]*180/np.pi, ant_lat[i]*180/np.pi)

    return ant_nam, ant_sta, ant_lon, ant_lat



def _extract_pointing_data(ms_name):
    antenna_ha = None
    antenna_dec = None

    pnttab = ctables.table(ms_name+'::POINTING', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    # This is the pointing direction at pnt_time, in radians in AZ EL
    pnt_dir = pnttab.getcol('DIRECTION')
    pnt_time = pnttab.getcol('TIME')
    pnttab.close()

    teltab = ctables.table()


    return pnt_time, antenna_ha, antenna_dec


def _extract_phase_gains(ms_name, pol_state, phase_gain_tol):
    time_vis = None
    phase_gains = None
    return time_vis, phase_gains
