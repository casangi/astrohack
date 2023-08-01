import numpy as np
from matplotlib import pyplot as plt

from casacore import tables as ctables
from astropy.coordinates import SkyCoord, CIRS
from astropy.time import Time
import astropy.units as units
from astropy.coordinates import EarthLocation
import xarray as xr

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._tools import _casa_time_to_mjd
from astrohack._utils._conversion import _convert_unit
from astrohack._utils._constants import figsize, twopi, fontsize
from astrohack._utils._dio import _write_meta_data


def _extract_antenna_data(fname, extract_locit_parms):
    """
    Extract antenna information from the ANTENNA sub table of the cal table
    Args:
        fname: Caller
        extract_locit_parms: input_parameters to extract_locit
    Returns:
    Antenna dictionary
    """
    logger = _get_astrohack_logger()
    cal_table = extract_locit_parms['cal_table']
    ant_table = ctables.table(cal_table + '::ANTENNA', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    ant_off = ant_table.getcol('OFFSET')
    ant_pos = ant_table.getcol('POSITION')
    ant_mnt = ant_table.getcol('MOUNT')
    ant_nam = ant_table.getcol('NAME')
    ant_sta = ant_table.getcol('STATION')
    ant_typ = ant_table.getcol('TYPE')
    ant_table.close()

    n_ant_orig = ant_off.shape[0]
    ant_pos_corrected = ant_pos + ant_off
    ant_rad = np.sqrt(ant_pos_corrected[:, 0] ** 2 + ant_pos_corrected[:, 1] ** 2 + ant_pos_corrected[:, 2] ** 2)
    ant_lat = np.arcsin(ant_pos_corrected[:, 2] / ant_rad)
    ant_lon = -np.arccos(ant_pos_corrected[:, 0] / (ant_rad * np.cos(ant_lat)))

    if extract_locit_parms['ant'] == 'all':
        ant_list = ant_nam
    else:
        ant_list = extract_locit_parms['ant']

    ant_dict = {}
    error = False
    for i_ant in range(n_ant_orig):
        this_name = ant_nam[i_ant]
        if this_name in ant_list:
            if ant_mnt[i_ant] != 'ALT-AZ':
                logger.error(f'[{fname}]: Antenna {this_name} has a non supported mount type: {ant_mnt[i_ant]}')
                error = True
            if ant_typ[i_ant] != 'GROUND-BASED':
                error = True
                logger.error(f'[{fname}]: Antenna {this_name} is not ground based which is currently not supported')
            if error:
                pass
            else:
                antenna = {'id': i_ant, 'name': this_name, 'station': ant_sta[i_ant],
                           'geocentric_position': ant_pos[i_ant], 'longitude': ant_lon[i_ant],
                           'latitude': ant_lat[i_ant], 'radius': ant_rad[i_ant], 'offset': ant_off[i_ant]}
                ant_dict[i_ant] = antenna

    if error:
        msg = f'[{fname}]: Unsupported antenna characteristics'
        logger.error(msg)
        raise Exception(msg)

    extract_locit_parms['ant_dict'] = ant_dict
    extract_locit_parms['full_antenna_list'] = ant_nam


def _extract_spectral_info(fname, extract_locit_parms):
    """
    Extract spectral information from the SPECTRAL_WINDOW sub table of the cal table
    Args:
        fname: Caller
        extract_locit_parms: input_parameters to extract_locit

    Returns:
    DDI dictionary
    """
    logger = _get_astrohack_logger()
    cal_table = extract_locit_parms['cal_table']
    spw_table = ctables.table(cal_table+'::SPECTRAL_WINDOW', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    ref_freq = spw_table.getcol('REF_FREQUENCY')
    n_chan = spw_table.getcol('NUM_CHAN')
    bandwidth = spw_table.getcol('CHAN_WIDTH')
    spw_table.close()
    n_ddi = len(ref_freq)
    error = False
    if extract_locit_parms['ddi'] == 'all':
        ddi_list = range(n_ddi)
    else:
        if isinstance(extract_locit_parms['ddi'], int):
            ddi_list = [extract_locit_parms['ddi']]
        else:
            ddi_list = extract_locit_parms['ddi']

    ddi_dict = {}
    for i_ddi in ddi_list:
        if n_chan[i_ddi] != 1:
            error = True
            msg = f'[{fname}]: DDI {i_ddi} has {n_chan[i_ddi]}, which is not supported'
            logger.error(msg)
        else:
            ddi_dict[i_ddi] = {'id': i_ddi, 'frequency': ref_freq[i_ddi], 'bandwidth': bandwidth[i_ddi]}

    if error:
        msg = f'[{fname}]: Unsupported DDI characteristics'
        logger.error(msg)
        raise Exception(msg)
    extract_locit_parms['ddi_dict'] = ddi_dict


def _extract_source_and_telescope(fname, extract_locit_parms):
    """
    Extract source and telescope  information from the FIELD and OBSERVATION sub tables of the cal table
    Args:
        fname: Caller
        extract_locit_parms: input_parameters to extract_locit

    Returns:
    Writes dict to a json file
    """
    cal_table = extract_locit_parms['cal_table']
    basename = extract_locit_parms['locit_name']
    src_table = ctables.table(cal_table+'::FIELD', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    src_ids = src_table.getcol('SOURCE_ID')
    phase_center_j2000 = src_table.getcol('PHASE_DIR')[:, 0, :]
    src_name = src_table.getcol('NAME')
    src_table.close()
    n_src = len(src_ids)

    phase_center_j2000[:, 0] = np.where(phase_center_j2000[:, 0] < 0, phase_center_j2000[:, 0]+twopi,
                                        phase_center_j2000[:, 0])

    obs_table = ctables.table(cal_table+'::OBSERVATION', readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    time_range = _casa_time_to_mjd(obs_table.getcol('TIME_RANGE')[0])
    telescope_name = obs_table.getcol('TELESCOPE_NAME')[0]
    obs_table.close()

    mid_time = Time((time_range[-1]+time_range[0])/2, scale='utc', format='mjd')

    astropy_j2000 = SkyCoord(phase_center_j2000[:, 0], phase_center_j2000[:, 1], unit=units.rad, frame='fk5')
    astropy_precessed = astropy_j2000.transform_to(CIRS(obstime=mid_time))
    phase_center_precessed = np.ndarray((n_src, 2))
    phase_center_precessed[:, 0] = astropy_precessed.ra
    phase_center_precessed[:, 1] = astropy_precessed.dec
    phase_center_precessed *= _convert_unit('deg', 'rad', 'trigonometric')

    src_dict = {}
    for i_src in range(n_src):
        src_id = int(src_ids[i_src])
        src_dict[src_id] = {'id': src_id, 'name': src_name[i_src], 'j2000': phase_center_j2000[i_src].tolist(),
                            'precessed': phase_center_precessed[i_src].tolist()}

    obs_dict = {'src_dict': src_dict, 'time_range': time_range.tolist(), 'telescope_name': telescope_name}
    if telescope_name == 'EVLA':
        tel_pos = EarthLocation.of_site('VLA')
    else:
        tel_pos = EarthLocation.of_site(telescope_name)
    obs_dict['array_center_geocentric'] = [tel_pos.x.value, tel_pos.y.value, tel_pos.z.value]
    obs_dict['array_center_lonlatrad'] = [tel_pos.lon.value, tel_pos.lat.value,
                                          np.sqrt(tel_pos.x**2+tel_pos.y**2+tel_pos.z**2).value]

    _write_meta_data("/".join([basename, ".observation_info"]), obs_dict)
    extract_locit_parms['telescope_name'] = telescope_name
    return telescope_name, n_src


def _extract_antenna_phase_gains(fname, extract_locit_parms):
    """
    Extract antenna based phase gains from the cal table
    Args:
        fname: Caller
        extract_locit_parms: input_parameters to extract_locit

    Returns:
    Reference antenna
    """
    logger = _get_astrohack_logger()
    cal_table = extract_locit_parms['cal_table']
    basename = extract_locit_parms['locit_name']
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
            logger.warning(f'[{fname}]: The best reference Antenna only covers {100*fraction_best:.1f}% of the data')
        for i_refant in range(n_refant):
            if i_refant != i_best_ant:
                logger.info(f'[{fname}]: Discarding gains derived with antenna '
                            f'{extract_locit_parms["full_antenna_list"][ref_antennas[i_refant]]}'
                            f' as reference ({100*counts[i_refant]/n_gains:.2f}% of the data)')
                sel_refant = antenna2 != ref_antennas[i_refant]
                antenna2 = antenna2[sel_refant]
                antenna1 = antenna1[sel_refant]
                gain_time = gain_time[sel_refant]
                gains = gains[sel_refant]
                fields = fields[sel_refant]
                spw_id = spw_id[sel_refant]
        ref_antenna = ref_antennas[i_best_ant]
    else:
        # No data to discard we can go on and compute the phase gains
        ref_antenna = ref_antennas[0]

    used_sources = []
    extract_locit_parms['reference_antenna'] = extract_locit_parms['full_antenna_list'][ref_antenna]
    phase_gains = np.angle(gains)
    for ant_id, antenna in extract_locit_parms['ant_dict'].items():
        ant_sel = antenna1 == ant_id
        ant_time = gain_time[ant_sel]
        ant_field = fields[ant_sel]
        ant_phase_gains = phase_gains[ant_sel]
        ant_spw_id = spw_id[ant_sel]
        if ant_id == ref_antenna:
            antenna['reference'] = True
        else:
            antenna['reference'] = False

        for ddi_id, ddi in extract_locit_parms['ddi_dict'].items():
            this_ddi_xds = xr.Dataset()
            ddi_sel = ant_spw_id == ddi_id
            coords = {"time": ant_time[ddi_sel]}
            this_ddi_xds.assign_coords(coords)
            this_ddi_xds['PHASE_GAINS'] = xr.DataArray(ant_phase_gains[ddi_sel], dims=('time', 'chan', 'pol'))
            this_ddi_xds['FIELD_ID'] = xr.DataArray(ant_field[ddi_sel], dims='time')
            this_ddi_xds.attrs['frequency'] = ddi['frequency']
            this_ddi_xds.attrs['bandwidth'] = ddi['bandwidth']
            outname = "/".join([basename, 'ant_'+antenna['name'], f'ddi_{ddi["id"]}'])
            this_ddi_xds.to_zarr(outname, mode="w", compute=True, consolidated=True)
            used_sources.extend(ant_field[ddi_sel])
        _write_meta_data("/".join([basename, 'ant_'+antenna['name'], ".antenna_info"]), antenna)
    extract_locit_parms['used_sources'] = np.unique(np.array(used_sources))
    return


def _plot_source_table(filename, src_dict, label=True, precessed=False, obs_midpoint=None, display=True,
                       figure_size=figsize, dpi=300):
    logger = _get_astrohack_logger()
    n_src = len(src_dict)
    radec = np.ndarray((n_src, 2))
    name = []
    if precessed:
        if obs_midpoint is None:
            msg = 'Observation midpoint is missing'
            logger.error(msg)
            raise Exception(msg)
        coorkey = 'precessed'
        time = Time(obs_midpoint, format='mjd')
        title = f'Coordinates precessed to {time.iso}'
    else:
        coorkey = 'j2000'
        title = 'J2000 reference frame'

    for i_src, src in src_dict.items():
        radec[int(i_src)] = src[coorkey]
        name.append(src['name'])

    if figure_size is None or figure_size == 'None':
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figure_size)
    radec[:, 0] *= _convert_unit('rad', 'hour', 'trigonometric')
    radec[:, 1] *= _convert_unit('rad', 'deg', 'trigonometric')

    if label:
        for i_src in range(n_src):
            ax.plot(radec[i_src, 0], radec[i_src, 1], marker='+', ls='', color='red')
            ax.text(radec[i_src, 0]+0.05, radec[i_src, 1], name[i_src], fontsize=.8*fontsize, ha='left', va='center',
                    rotation=20)
    else:
        ax.plot(radec[:, 0], radec[:, 1], marker='+', ls='', color='red')
    ax.set_xlim([-0.5, 24.5])
    ax.set_ylim([-95, 95])
    ax.set_xlabel('Right Ascension [h]')
    ax.set_ylabel('Declination [\u00b0]')

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename, dpi=dpi)
    if not display:
        plt.close()
    return


def _plot_antenna_table(filename, ant_dict, array_center, stations=True, display=True, figure_size=figsize, dpi=300):
    if figure_size is None or figure_size == 'None':
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figure_size)

    rad2deg = _convert_unit('rad', 'deg', 'trigonometric')
    # ax.plot(array_center[0], array_center[1], marker='x', color='blue')
    title = 'Antenna positions during observation'
    for antenna in ant_dict.values():
        long = antenna['longitude']*rad2deg
        lati = antenna['latitude']*rad2deg
        ax.plot(long, lati, marker='+', color='black')
        text = f'  {antenna["name"]}'
        if stations:
            text += f'@{antenna["station"]}'
        ax.text(long, lati, text, fontsize=fontsize, ha='left', va='center', rotation=0)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    x_half, x_mid = (x_lim[1] - x_lim[0])/2, (x_lim[1] + x_lim[0]) / 2
    y_half, y_mid = (y_lim[1] - y_lim[0])/2, (y_lim[1] + y_lim[0]) / 2
    if x_half > y_half:
        y_lim = [y_mid-x_half, y_mid+x_half]
    else:
        x_lim = [x_mid-y_half, x_mid+y_half]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('Longitude [\u00b0]')
    ax.set_ylabel('Latitude [\u00b0]')

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename, dpi=dpi)
    if not display:
        plt.close()
    return
