import numpy as np

from casacore import tables as ctables
from astropy.coordinates import EarthLocation, AltAz, HADec, SkyCoord
from astropy.time import Time
import astropy.units as units
import xarray as xr
import time

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._tools import _casa_time_to_mjd, _altaz_to_hadec


def _extract_antenna_data(fname, cal_table):
    logger = _get_astrohack_logger()

    ant_table = ctables.table(cal_table + '::ANTENNA', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
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
        msg = f'[{fname}]: Unsupported antenna characteristics'
        logger.error(msg)
        raise Exception(msg)

    ant_pos_corrected = ant_pos + ant_off
    ant_rad = np.sqrt(ant_pos_corrected[:, 0] ** 2 + ant_pos_corrected[:, 1] ** 2 + ant_pos_corrected[:, 2] ** 2)
    ant_lat = np.arcsin(ant_pos_corrected[:, 2] / ant_rad)
    ant_lon = -np.arccos(ant_pos_corrected[:, 0] / (ant_rad * np.cos(ant_lat)))

    ant_dict = {'n_ant': n_ant, 'name': ant_nam, 'station': ant_sta, 'longitude': ant_lon, 'latitude': ant_lat,
                'radius': ant_rad, 'position': ant_pos, 'offset': ant_off, 'corrected': ant_pos_corrected}
    return ant_dict


def _extract_spectral_info(fname, cal_table):
    logger = _get_astrohack_logger()
    spw_table = ctables.table(cal_table+'::SPECTRAL_WINDOW', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    ref_freq = spw_table.getcol('REF_FREQUENCY')
    n_chan = spw_table.getcol('NUM_CHAN')
    bandwidth = spw_table.getcol('CHAN_WIDTH')
    spw_table.close()
    n_ddi = len(ref_freq)
    error = False
    for i_ddi in range(n_ddi):
        if n_chan[i_ddi] != 1:
            error = True
            msg = f'[{fname}]: DDI {i_ddi} has {n_chan[i_ddi]}, which is not supported'
            logger.error(msg)
    if error:
        msg = f'[{fname}]: Unsupported DDI characteristics'
        logger.error(msg)
        raise Exception(msg)
    ddi_dict = {'n_ddi': n_ddi, 'frequencies': ref_freq, 'bandwidth': bandwidth}
    return ddi_dict


def _extract_source_info(fname, cal_table):
    src_table = ctables.table(cal_table+'::FIELD', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    src_id = src_table.getcol('SOURCE_ID')
    phase_center_j2000 = src_table.getcol('PHASE_DIR')
    src_name = src_table.getcol('NAME')
    src_table.close()
    n_src = len(src_id)

    src_dict = {'n_src': n_src, 'id': src_id, 'name': src_name, 'radec_j2000': phase_center_j2000}
    return src_dict


def _extract_antenna_phase_gains(fname, cal_table, ant_dict, ddi_dict):
    logger = _get_astrohack_logger()
    main_table = ctables.table(cal_table, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    antenna1 = main_table.getcol('ANTENNA1')
    antenna2 = main_table.getcol('ANTENNA2')
    gain_time = main_table.getcol('TIME')
    gains = main_table.getcol('CPARAM')
    fields = main_table.getcol('FIELD_ID')
    spw_id = main_table.getcol('SPECTRAL_WINDOW_ID')
    main_table.close()
    n_gains = len(gains)

    ref_antennas, counts = np.unique(antenna2, return_counts=True)
    n_refant = len(ref_antennas)
    if n_refant > 1:
        i_best_ant = np.argmax(counts)
        fraction_best = counts[i_best_ant]/n_gains
        if fraction_best < 0.5:
            logger.warning(f'[{fname}]: The best reference Antenna only covers {100*fraction_best}% of the data')
        for i_refant in range(n_refant):
            if i_refant != i_best_ant:
                logger.info(f'[{fname}]: Discarding gains derived with antenna {ant_dict["name"][ref_antennas[i_refant]]} as reference')
                sel_refant = antenna2 != ref_antennas[i_refant]
                antenna2 = antenna2[sel_refant]
                antenna1 = antenna1[sel_refant]
                gain_time = gain_time[sel_refant]
                gains = gains[sel_refant]
                fields = fields[sel_refant]
                spw_id = spw_id[sel_refant]
    else:
        # No data to discard we can go on and compute the phase gains
        pass
    phase_gains = np.angle(gains)
    global_dict = {}
    for i_ant in range(ant_dict['n_ant']):
        this_ant_dict = {}
        ant_sel = antenna1 == i_ant
        ant_time = gain_time[ant_sel]
        ant_field = fields[ant_sel]
        ant_phase_gains = phase_gains[ant_sel]
        ant_spw_id = spw_id[ant_sel]
        for i_ddi in range(ddi_dict['n_ddi']):
            this_ddi_xds = xr.Dataset()
            ddi_sel = ant_spw_id == i_ddi
            coords = {"time": ant_time[ddi_sel]}
            this_ddi_xds.assign_coords(coords)
            this_ddi_xds['PHASE_GAINS'] = xr.DataArray(ant_phase_gains[ddi_sel], dims=('time', 'chan', 'pol'))
            this_ddi_xds['FIELD_ID'] = xr.DataArray(ant_field[ddi_sel], dims='time')
            this_ant_dict[f'DDI_{i_ddi}'] = this_ddi_xds
        global_dict[ant_dict['name'][i_ant]] = this_ant_dict

    return global_dict


