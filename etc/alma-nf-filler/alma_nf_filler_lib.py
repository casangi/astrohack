from asdm import *
import numpy as np
import xarray as xr
import scipy
from astrohack.antenna.telescope import Telescope
from matplotlib import pyplot as plt
from scipy import interpolate
import time as timer

RAD2DEG = 180/np.pi
DAY2SEC = 86400
VERSION = '0.0.1'

##############################
#   Direct ASDM handling   ###
##############################


def asdm_to_holog(asdm_name: str,
                  holog_name: str,
                  int_time: float = None,
                  verbose: bool = False,
                  cal_amp: str = 'linterp',
                  cal_pha: str = 'linterp',
                  cal_cycle: int = 3,
                  plot_cal: bool = False,
                  save_cal: bool = False):
    """
    Master filler routine that controls the work flow
    Args:
        asdm_name: Near field ASDM file + its path
        holog_name: Nome of the .holog.zarr file to be created without extension
        int_time: Integration time bin, if None it is set to the largest time bin between pointing and total power
        verbose: print processing detailed messages
        cal_amp: Amplitude calibration fitting, None means no amplitude calibration
        cal_pha: Phase calibration fitting, None means no phase calibration
        cal_cycle: Number of subscans in a Calibration, data cycle, seems to always be 3
        plot_cal: Plot calibration to png files for inspection
        save_cal: Save calibrated total power data to a .zarr file

    Returns:
    None
    """
    start = timer.time()
    print(f'Processing {asdm_name} to {holog_name}.holog.zarr...\n')
    
    asdm_object = _open_asdm(asdm_name)
    meta_dict = get_asdm_metadata(asdm_object)
    pnt_info = get_pnt_from_asdm(asdm_object, verbose)
    tp_info = get_tp_from_asdm(asdm_object, meta_dict['holo'], verbose)

    calibrated_data = calibrate_tp_data(tp_info, cal_amp, cal_pha, cal_cycle,
                                        plot_cal, holog_name, verbose)
    if save_cal:
        calibrated_data.to_zarr(holog_name+'.cal.zarr', mode='w', compute=True, consolidated=True)

    combined_data = combine_data(meta_dict, pnt_info, calibrated_data, int_time,
                                 verbose)
    export_data(asdm_name, combined_data, meta_dict, holog_name, int_time,
                verbose)

    el_time = timer.time()-start
    print(f'\nFinished processing ({el_time:.2f} seconds)')
    

def _open_asdm(asdm_name):
    """
    This routine interacts with the ASDM bindings to render the data accessible
    Args:
        asdm_name: Near field ASDM file + its path

    Returns:
    ASDM object with the data now accessible
    """
    parser = ASDMParseOptions()
    parser.asALMA()
    parser.loadTablesOnDemand(True)

    asdm_object = ASDM()
    asdm_object.setFromFile(asdm_name, parser)
    return asdm_object


#######################
# Retrieve metadata ###
#######################

def get_asdm_metadata(asdm_object):
    """
    Retrieve metadata relevant for near field holography from the ASDM object
    Args:
        asdm_object: The ASDM object

    Returns:
    A dictionary containing the relevant metadata
    """
    meta_dict = {'holo': _get_holography_info(asdm_object),
                 'spw': _get_frequency_info(asdm_object),
                 'ant': _get_antenna_info(asdm_object),
                 'flagged_times': _get_flag_info(asdm_object)}
    return meta_dict


def _get_frequency_info(asdm_object):
    """
    Retrieve spectral window metadata from the ASDM
    Args:
        asdm_object: ASDM object

    Returns:
    dictionary with relevant frequency metadata
    """
    spw_info = asdm_object.spectralWindowTable().get()[0]
    
    spw_dict = {'nchan': spw_info.numChan(),
                'frequency': np.array([spw_info.refFreq().get()])}
    return spw_dict


def _get_antenna_info(asdm_object):
    """
    Get antenna metadata from the ASDM
    Args:
        asdm_object: ASDM object

    Returns:
    Dictionary with relevant antenna metadata
    """
    ant_info = asdm_object.antennaTable().get()[0]

    tel_dict = {'antenna': ant_info.name(),
                'telescope': 'ALMA'}

    return tel_dict


def _get_holography_info(asdm_object):
    """
    Retrieve near field holography metadata from the ASDM object
    Args:
        asdm_object: ASDM object

    Returns:
    Dictionary with near field holography metadata
    """
    try:
        holo_info = asdm_object.holographyTable().get()[0]
    except:
        raise Exception('ASDM is not a ALMA holography')

    corr_axis = []
    for item in holo_info.type():
        corr_axis.append(str(item))
    
    holo_dict = {'focus': holo_info.focus().get(),
                 'ncorr': holo_info.numCorr(),
                 'corr_axis': np.array(corr_axis)}
    
    return holo_dict


def _get_flag_info(asdm_object):
    """
    Retrieve flagging data from the ASDM object
    Args:
        asdm_object: ASDM object

    Returns:
    List with all the flagging times
    """
    flag_table = asdm_object.flagTable().get()
    nflags = len(flag_table)

    flag_times = np.ndarray([nflags, 2])
    for i_row, flag_row in enumerate(flag_table):
        flag_times[i_row, 0] = flag_row.startTime().getMJD()
        flag_times[i_row, 1] = flag_row.endTime().getMJD()

    return flag_times


###########################
#   Retrieve pointing   ###
###########################

def get_pnt_from_asdm(asdm_object, verbose):
    """
    Retrieve pointing data from the ASDM object and excludes pointing outliers
    Args:
        asdm_object: ASDM object
        verbose: Print processing messages?

    Returns:
    Dictionary with filtered pointing data
    """
    pnt_table = asdm_object.pointingTable().get()
    num_samples = 0
    pnt_time = np.ndarray((0, ))
    pnt_scan = np.ndarray((0, ))
    pnt_dir = np.ndarray((0, 2))
    pnt_enc = np.ndarray((0, 2))
    pnt_tgt = np.ndarray((0, 2))
    pnt_off = np.ndarray((0, 2))
    
    for irow, pnt_row in enumerate(pnt_table):
        row_time, row_scan, row_dir, row_enc, row_tgt, row_off, n_samp = _get_pnt_from_row(pnt_row, irow)
        num_samples += n_samp
        
        pnt_time = np.concatenate([pnt_time, row_time])
        pnt_scan = np.concatenate([pnt_scan, row_scan])
        pnt_dir = np.concatenate([pnt_dir, row_dir])
        pnt_enc = np.concatenate([pnt_enc, row_enc])
        pnt_tgt = np.concatenate([pnt_tgt, row_tgt])
        pnt_off = np.concatenate([pnt_off, row_off])

    if verbose:    
        print("Total number of pointing samples: ", num_samples)
    pnt_dict = {'time': pnt_time,
                'scan': pnt_scan,
                'direction': pnt_dir,
                'encoder': pnt_enc,
                'target': pnt_tgt,
                'offset': pnt_off,           
                }
    l_axis = np.cos(pnt_dict['target'][:, 1]) * np.sin(pnt_dict['target'][:, 0] - pnt_dict['direction'][:, 0])
    m_axis = np.sin(pnt_dict['target'][:, 1]) * np.cos(pnt_dict['direction'][:, 1]) - np.cos(pnt_dict['target'][:, 1])\
        * np.sin(pnt_dict['direction'][:, 1]) * np.cos(pnt_dict['target'][:, 0] - pnt_dict['direction'][:, 0])

    pnt_dict['l'] = l_axis
    pnt_dict['m'] = m_axis
     
    pnt_dict = _filter_axes(pnt_dict, tol=0.0001, ref_axis='l')
    pnt_dict = _filter_axes(pnt_dict, tol=0.0001, ref_axis='m')

    pnt_dict['nsamp'] = pnt_dict['time'].shape[0]

    if verbose:
        print("Total number of valid pointing samples: ", pnt_dict['nsamp'], '\n')
    
    return pnt_dict


def _filter_axes(pnt_dict, tol=0.0001, ref_axis='direction'):
    """
    Removes outliers based on a standard deviation rate of change recursevely.
    Args:
        pnt_dict: The unfiltered pointing dictionary
        tol: The standard deviation change gauge
        ref_axis: Axis over which to apply the filtering
    Returns:
    Outlier filtered pointing dictionary
    """
    in_axis = pnt_dict[ref_axis]
    in_std = np.nanstd(in_axis)
    outlier = np.nanmax(np.abs([np.nanmin(in_axis), np.nanmax(in_axis)]))
    selection = np.abs(in_axis) != outlier
    ou_axis = in_axis[selection]
    ou_std = np.nanstd(ou_axis)
    std_diff = np.abs(in_std-ou_std)
    
    if std_diff < tol:
        return pnt_dict
    else:
        ou_dict = {}
        for key, value in pnt_dict.items():
            ou_dict[key] = pnt_dict[key][selection]
        return _filter_axes(ou_dict, tol=tol, ref_axis=ref_axis)


def _get_pnt_from_row(pnt_row, irow):
    """
    get pointing information from an ASDM's pointing row (subscan)
    Args:
        pnt_row: the ASDM's pointing row
        irow: rows number

    Returns:
    numpy arrays for time, scan, direction, encoder, target and offset, also number of samples
    """
    time_int = pnt_row.timeInterval()
    start_MJD = time_int.start().getMJD()
    # Where does this 1e9 comes from???
    duration = time_int.duration().get()/1e9/DAY2SEC 
    n_samp = pnt_row.numSample()
    time_int = duration/n_samp
    # subtracting a time_int guaranties a correct time step
    stop_MJD = start_MJD + duration-time_int 
    row_time = np.linspace(start_MJD, stop_MJD, num=n_samp)
    row_scan = np.full(n_samp, irow+1)
    row_dir = _asdm_angle_tuple_to_numpy(pnt_row.pointingDirection())
    row_enc = _asdm_angle_tuple_to_numpy(pnt_row.encoder())
    row_tgt = _asdm_angle_tuple_to_numpy(pnt_row.target())
    row_off = _asdm_angle_tuple_to_numpy(pnt_row.offset())
    return row_time, row_scan, row_dir, row_enc, row_tgt, row_off, n_samp


def _asdm_angle_tuple_to_numpy(letuple):
    """
    Convert from ASDM tuple of angle quantities to a numpy array of float angles in radians
    Args:
        letuple: the ASDM row tuple

    Returns:
    numpy array of float angles in radians
    """
    # 2D tuple assumed here
    n_long = len(letuple)
    n_wide = len(letuple[0])
    np_data = np.ndarray((n_long, n_wide))
    for i_long in range(n_long):
        for i_wide in range(n_wide):
            np_data[i_long, i_wide] = letuple[i_long][i_wide].get()
    return np_data


##########################
#   Retrieve tp data   ###
##########################

def get_tp_from_asdm(asdm_object, holo_info, verbose):
    """
    Retrieve total power data from the ASDM rows, also converts correlations to Intensity amplitude and phase.
    Args:
        asdm_object: ASDM object
        holo_info: Holography relevant metadata
        verbose: print processing messages?

    Returns:
    dictionary with the total power data
    """
    if verbose:
        print('Retrieving Total Power Table...')
        start = timer.time()
    tp_table = asdm_object.totalPowerTable().get()
    if verbose:
        el_time = timer.time()-start
        print(f'Done retrieving Total Power Table, took {el_time:.2f} seconds')
    num_samples = len(tp_table)

    tp_time = np.ndarray((num_samples, ))
    tp_scan = np.ndarray((num_samples, ))
    tp_r2 = np.ndarray((num_samples, ))
    tp_qr = np.ndarray((num_samples, ))
    tp_rs = np.ndarray((num_samples, ))
    
    idict = _get_corr_indexes(holo_info, ['R2', 'QR', 'RS'])

    for i_row, tp_row in enumerate(tp_table):
        tp_time[i_row] = tp_row.time().getMJD()
        tp_scan[i_row] = tp_row.subscanNumber()
        tp_r2[i_row] = tp_row.floatData()[0][0][idict['R2']]
        tp_qr[i_row] = tp_row.floatData()[0][0][idict['QR']]
        tp_rs[i_row] = tp_row.floatData()[0][0][idict['RS']]

    if verbose:
        print("Total number of Total power samples: ", num_samples, '\n')

    # Real and imaginary parts from correlations
    tp_s = tp_rs/tp_r2
    tp_q = tp_qr/tp_r2

    # Amplitude = sqrt(Q2 + S2)
    tp_amp = np.sqrt(tp_s**2+tp_q**2)
    
    # Phase = arc-tangent(Q/S)
    tp_pha = np.arctan(tp_q/tp_s)

    tp_dict = {'time': tp_time,
               'scan': tp_scan,
               'ref':  tp_r2,
               'amp':  tp_amp,
               'pha':  tp_pha,
               'nsamp': num_samples}
    return tp_dict


def _get_corr_indexes(holo_info, wanted_corr):
    """
    Get the indexes of the wanted correlations from the holography metadata
    Args:
        holo_info: Holography metadata
        wanted_corr: The correlation for which we want indexes
    Returns:
    Dictionary of correlation indexes
    """
    idict = {}
    for corr in wanted_corr:
        if corr in holo_info['corr_axis']:
            idict[corr] = list(holo_info['corr_axis']).index(corr)
        else:
            Exception(f'ERROR: {corr} correlation not present in data')
    return idict


#####################
#   Calibration   ###
#####################

def calibrate_tp_data(tp_info, cal_amp, cal_pha, cal_cycle, plot_cal,
                      holog_name, verbose):
    """
    Calibrate the data bade on a simple scheme derived from:
    Near-Field Radio Holography of Large Reflector Antennas
    DOI: 10.1109/MAP.2007.4395293 · Source: IEEE Xplore

    Args:
        tp_info: Total Power dictionary
        cal_amp: al_amp: Amplitude calibration fitting, None means no amplitude calibration
        cal_pha: Phase calibration fitting, None means no phase calibration
        cal_cycle: Number of subscans in a Calibration, data cycle, seems to always be 3
        plot_cal: Plot calibration to png files for inspection
        holog_name: Name of the output .holog.zarr file (used to determine the plot filenames)
        verbose: print processing messages?

    Returns:
    Enriched total power Xarray dataset with the calibrated data (when aplicable)
    """

    cal_data, data = _separate_cal_from_data(tp_info, cal_cycle)

    _amplitude_calibration(cal_data, data, cal_amp, plot_cal, holog_name, verbose)
    if verbose:
        print()
    _phase_calibration(cal_data, data, cal_pha, plot_cal, holog_name, verbose)
    if verbose:
        print()
            
    return data


def _separate_cal_from_data(tp_info, cal_cycle):
    """
    Separate the total power dictionary onto two Xarray datasets, one with the calibration data, the other with the
    holography data
    Args:
        tp_info: total power dictionary
        cal_cycle: Number of subscans in a calibration cycle, usually 3 (cal, obs, obs, cal, obs, obs)

    Returns:
    Calibration data, and holography data Xarray datasets.
    """
    time = tp_info['time']
    scan = tp_info['scan'] 
    amp = tp_info['amp']
    pha = tp_info['pha']
    ref = tp_info['ref']

    cal_time, cal_amp, cal_pha = [], [], []
    dat_time, dat_amp, dat_pha, dat_ref = [], [], [], []

    for i_row in range(tp_info['nsamp']):
        if int(scan[i_row]) % cal_cycle == 1:
            cal_time.append(time[i_row])
            cal_amp.append(amp[i_row])
            cal_pha.append(pha[i_row])
        else:
            dat_time.append(time[i_row])
            dat_amp.append(amp[i_row])
            dat_pha.append(pha[i_row])
            dat_ref.append(ref[i_row])

    cal_xds = _create_xds(cal_time, cal_amp, cal_pha, True)
    dat_xds = _create_xds(dat_time, dat_amp, dat_pha, False, ref=dat_ref)

    return cal_xds, dat_xds


def _create_xds(time, amp, pha, is_cal, ref=None):
    """
    Create a Xarray dataset from the given lists
    Args:
        time: Total power times
        amp: Total power Amplitudes
        pha: Total power Phases
        is_cal: Is this a calibration data set
        ref: Total power reference power

    Returns:
    An Xarray dataset with the input lists
    """
    coords = {"time": np.array(time)}
    
    xds = xr.Dataset()
    xds = xds.assign_coords(coords)
    xds["AMPLITUDE"] = xr.DataArray(np.array(amp), dims=["time"])
    xds["PHASE"] = xr.DataArray(np.array(pha), dims=["time"])

    if ref is not None:
        xds['REFERENCE'] = xr.DataArray(np.array(ref), dims=["time"])

    xds.attrs['calibration'] = is_cal
    if not is_cal:
        xds.attrs['phase_cal'] = False
        xds.attrs['amplitude_cal'] = False
    return xds
    

def _solve_calibration(cal_time, cal_values, data_time, cal_type):
    """
    Fits the calibration data using the correct cal_type
    Args:
        cal_time: The calibration time axis
        cal_values: The quantity over which to find an interpolation
        data_time: The data sampling times
        cal_type: The type of calibration fitting to be performed

    Returns:
    The calibration value sinterpolated to the data sampling times
    """
    if cal_type == 'none':
        return None
    elif cal_type in cal_methods.keys():
        return cal_methods[cal_type](cal_time, cal_values, data_time)
    else:
        raise Exception(f"Unknown calibration solving algorithm {cal_type}")

    
def _spline_fit(cal_time, cal_data, data_time):
    """
    Fit calibration data to a spline (Very bad fits)
    Args:
        cal_time: Calibration time samnpling
        cal_data: Calibration data
        data_time: Observation time samnpling

    Returns:
    The fitted calibration interpolated over the data time sampling
    """
    spl_coeff = interpolate.splrep(cal_time, cal_data)
    spl_val = interpolate.splev(data_time, spl_coeff)
    return spl_val


def _mean_fit(cal_time, cal_data, data_time):
    """
    Fit calibration data to a simple mean of the calibration data (too rough)
    Args:
        cal_time: Calibration time samnpling (Added for interface compatibility)
        cal_data: Calibration data
        data_time: Observation time samnpling

    Returns:
    The fitted calibration interpolated over the data time sampling
    """
    mean_val = np.nanmean(cal_data)
    return np.full_like(data_time, mean_val)


def _linear_regression_fit(cal_time, cal_data, data_time):
    """
    Fit calibration data with a linear regression (Too rough)
    Args:
        cal_time: Calibration time samnpling
        cal_data: Calibration data
        data_time: Observation time samnpling

    Returns:
    The fitted calibration interpolated over the data time sampling
    """
    system = np.empty([cal_time.shape[0], 2])
    system[:, 0] = cal_time
    system[:, 1] = 1.0
    
    regression = np.linalg.lstsq(system, cal_data, rcond=None)
    vec = regression[0]
    solution = vec[0]*data_time + vec[1]
    return solution


def _linear_interpolation_fit(cal_time, cal_data, data_time):
    """
    Fit calibration data using a simple linear interpolation in between gaps (best results)
    Args:
        cal_time: Calibration time samnpling
        cal_data: Calibration data
        data_time: Observation time samnpling

    Returns:
    The fitted calibration interpolated over the data time sampling
    """
    func = interpolate.interp1d(cal_time, cal_data)
    return func(data_time)


def _square_interpolation_fit(cal_time, cal_data, data_time):
    """
    Fit calibration data to a quadratic spline (Very bad fits)
    Args:
        cal_time: Calibration time samnpling
        cal_data: Calibration data
        data_time: Observation time samnpling

    Returns:
    The fitted calibration interpolated over the data time sampling
    """
    func = interpolate.interp1d(cal_time, cal_data, kind='quadratic')
    return func(data_time)
    
    
"""
Dictionary of calibration methods used to overload functions
"""
cal_methods = {'linterp': _linear_interpolation_fit,
               'spline': _spline_fit,
               'mean': _mean_fit,
               'regression': _linear_regression_fit,
               'sqinterp': _square_interpolation_fit,
               }
CALIBRATION_OPTIONS = list(cal_methods.keys())
CALIBRATION_OPTIONS.append('none')


def _amplitude_calibration(cal, data, cal_type, plot_cal, holog_name, verbose):
    """
    Performs amplitude calibration by dividing data by the interpolated values of the calibration
    Args:
        cal: Calibration Xarray dataset
        data: Data Xarray dataset
        cal_type: Type of amplitude calibration to be performed
        plot_cal: Plot amplitude calibration to a png file?
        holog_name: Name of the output .holog.zarr file (used to determine plot filename)
        verbose: Print processing messages?

    Returns:
    The data Xarray dataset enriched with the calibrated amplitude 
    """
    start = timer.time()
    amp_sol = _solve_calibration(cal.time.values, cal['AMPLITUDE'].values,
                                 data.time.values, cal_type)
    
    # If amp_sol is None no amplitude calibration is to be performed
    if amp_sol is None:
        if verbose:
            print('Skipping Amplitude calibration')
        return
    if verbose:
        print('Calibrating Amplitude')
    
    amp = data['AMPLITUDE'].values.copy()

    # This leads to a calibration where amplitude ~1 towards the beam
    # center
    calibrated_amp = amp/amp_sol

    data['CALIBRATED_AMPLITUDE'] = xr.DataArray(calibrated_amp, dims=["time"])
    data['AMPLITUDE_SOLUTION'] = xr.DataArray(amp_sol, dims=["time"])
    data.attrs['amplitude_cal'] = True

    if verbose:
        el_time = timer.time()-start
        print(f'Amplitude calibration took: {el_time:.4f} seconds')

    if plot_cal:
        _plot_cal(amp_sol, cal, data, 'AMPLITUDE', holog_name, verbose)
    
    return data


def _phase_calibration(cal, data, cal_type, plot_cal, holog_name, verbose):
    """
    Performs phase calibration by subtracting interpolated values of the calibration from the data
    Args:
        cal: Calibration Xarray dataset
        data: Data Xarray dataset
        cal_type: Type of phase calibration to be performed
        plot_cal: Plot phase calibration to a png file?
        holog_name: Name of the output .holog.zarr file (used to determine plot filename)
        verbose: Print processing messages?

    Returns:
    The data Xarray dataset enriched with the calibrated phase 
    """
    start = timer.time()
    pha_sol = _solve_calibration(cal.time.values, cal['PHASE'].values,
                                 data.time.values, cal_type)
    
    # If pha_sol is None no phase calibration is to be performed
    if pha_sol is None:
        if verbose:
            print('Skipping phase calibration')
        return
    if verbose:
        print('Calibrating phase')

    pha = data['PHASE'].values.copy()
    
    calibrated_pha = pha-pha_sol
    calibrated_pha = np.where(calibrated_pha <= -np.pi/2, calibrated_pha+np.pi,
                              calibrated_pha)
    calibrated_pha = np.where(calibrated_pha >= np.pi/2, calibrated_pha-np.pi,
                              calibrated_pha)

    data['CALIBRATED_PHASE'] = xr.DataArray(calibrated_pha, dims=["time"])
    data['PHASE_SOLUTION'] = xr.DataArray(pha_sol, dims=["time"])
    data.attrs['phase_cal'] = True

    if verbose:
        el_time = timer.time()-start
        print(f'Phase calibration took: {el_time:.4f} seconds')

    if plot_cal:
        _plot_cal(pha_sol, cal, data, 'PHASE', holog_name, verbose)
    
    return data


def _plot_cal(sol, cal, data, cal_type, holog_name, verbose):
    """
    Plot calibration information to a png file
    Args:
        sol: The interpolated calibration solution
        cal: The calibration data Xarray dataset
        data: The actual observation Xarray datset
        cal_type: Is it phase or amplitude?
        holog_name: The base output name for the png file
        verbose: print processing messages?

    Returns:
    None
    """
    data_time = data.time.values.copy()
    cal_time = cal.time.values.copy()
    
    raw_data = data[cal_type].copy()
    cal = cal[cal_type].copy()
    caled_data = data[f'CALIBRATED_{cal_type}'].copy()

    # Zero time at the start of first calibration
    data_time -= cal_time[0]
    cal_time -= cal_time[0]

    # Time in minutes
    data_time *= 24*60
    cal_time *= 24*60

    if cal_type == 'PHASE':

        sol *= RAD2DEG
        raw_data *= RAD2DEG
        cal *= RAD2DEG
        caled_data *= RAD2DEG
        ylabel = 'Phase [DEG]'
    else:
        ylabel = 'Amplitude [Arb. units]'

    plt.title(f"{cal_type.lower().capitalize()} calibration")
    plt.plot(cal_time, cal, marker='.', ls='', color='red', label='Calibration')
    plt.plot(data_time, raw_data, marker='.', ls='', color='blue', label='Raw data')
    plt.plot(data_time, sol, marker='.', ls='', color='yellow',
             label='Cal. sol.')
    plt.plot(data_time, caled_data, marker='.', ls='', color='black',
             label='Cal.ed data')

    plt.xlabel('Time from start [m]')
    plt.ylabel(ylabel)

    plt.legend()

    fname = f'{holog_name}_{cal_type.lower()}_cal.png'
    if verbose:
        print(f'Saving {cal_type.lower()} calibration to {fname}')
    plt.savefig(fname, dpi=600)
    plt.clf()
    return


#####################
#   Data mixing   ###
#####################

def combine_data(meta_dict, pnt_info, tp_info, int_time, verbose):
    """
    This function combines the pointing data with the tp data while
    taking into account the flags, all work here assumes time is
    monotonically crescent
    Args:
        meta_dict: The metadata dictionary
        pnt_info: The pointing dictionary
        tp_info: The total power Xarray dataset
        int_time: The time bin size
        verbose: print processing messages?

    Returns:
    Matched tp and pnt data dictionary
    """
    start = timer.time()
    if verbose:
        print('Combining pointing and total power data...')
    time_axes, int_time = _get_time_axes(int_time, pnt_info, tp_info)

    new_pnt_info = _match_pnt_to_time(pnt_info, time_axes, meta_dict)
    new_tp_info = _match_tp_to_time(tp_info, time_axes, meta_dict)
    if verbose:
        print(f'Number of time matched samples: {new_tp_info["nsamp"]}')
    
    matched_info = _match_tp_and_pnt(new_tp_info, new_pnt_info)
    if verbose:
        print(f'Number of filtered samples: {matched_info["nsamp"]}')

    matched_info['integ_time'] = int_time
    if verbose:
        el_time = timer.time()-start
        print(f'Combination took {el_time:.2f} seconds\n')
    
    return matched_info
    
    
def _get_time_interval(int_time, pnt_time, tp_time):
    """
    Derive the best integration time based on tp and pnt time samplings
    Args:
        int_time: User input integration time
        pnt_time: Pointing time sampling
        tp_time: Total power time sampling

    Returns:
    The integration time bin
    """
    if int_time is None:
        tp_int = tp_time[1]-tp_time[0]
        pnt_int = pnt_time[1]-pnt_time[0]

        if tp_int > pnt_int:
            return tp_int
        else:
            return pnt_int
    else:
        return int_time


def _get_start_stop(pnt_time, tp_time):
    """
    Get the complete range of observation times
    Args:
        pnt_time: pointing time sampling
        tp_time: total power time sampling

    Returns:
    The observation start and stop times
    """
    if pnt_time[0] > tp_time[0]:
        start = pnt_time[0]
    else:
        start = tp_time[0]
    if pnt_time[-1] > tp_time[-1]:
        stop = tp_time[-1]
    else:
        stop = pnt_time[-1]
    return start, stop


def _get_time_axes(int_time, pnt_info, tp_info):
    """
    Derive the best time axis based on the integration time and time samplings
    Args:
        int_time: The user defined integration time
        pnt_info: the pointing information dictionary
        tp_info: The total power Xarray dataset

    Returns:
    The start and stop times of all samples and integration time
    """
    pnt_time = pnt_info['time']
    tp_time = tp_info['time'].values
    
    int_time = _get_time_interval(int_time, pnt_time, tp_time)
    start, stop = _get_start_stop(pnt_time, tp_time)

    total_int = stop-start
    n_time = int(np.ceil(total_int/int_time))

    final_time = start + (n_time-1)*int_time
    time_start = np.linspace(start, final_time, num=n_time)
    time_stop = time_start+int_time
    return np.array([time_start, time_stop]).T, int_time


def _match_pnt_to_time(pnt_info, time_axes, meta_dict):
    """
    Match pointing data to the new time axes
    Args:
        pnt_info: pointing info dictionary
        time_axes: The new time axes
        meta_dict: Metadata dictionary containing flags

    Returns:
    Time matched pointing data dictionary
    """
    flag_times = meta_dict['flagged_times']
    iflag = 0
    itime = 0
    the_shape = time_axes.shape

    pnt_dir = np.zeros(the_shape)
    pnt_off = np.zeros(the_shape)
    pnt_wei = np.zeros(the_shape)
    pnt_lm  = np.zeros(the_shape)
    
    for ipnt in range(pnt_info['nsamp']):
        pnt_time = pnt_info['time'][ipnt]
        if pnt_time > flag_times[iflag, 1]:
            if iflag == flag_times.shape[0]-1:
                pass
            else:
                iflag += 1
        
        while pnt_time > time_axes[itime, 1]:
            if itime == the_shape[0]-1:
                break
            else:
                itime += 1
        if pnt_time < time_axes[itime, 0]:
            continue
        elif flag_times[iflag, 0] <= pnt_time <= flag_times[iflag, 1]:
            continue
        
        pnt_dir[itime] += pnt_info['direction'][ipnt]
        pnt_off[itime] += pnt_info['offset'][ipnt]
        pnt_lm[itime, 0] += pnt_info['l'][ipnt]
        pnt_lm[itime, 1] += pnt_info['m'][ipnt]
        pnt_wei[itime] += 1.0
        
    with np.errstate(divide='ignore', invalid='ignore'):    
        pnt_dir /= pnt_wei
        pnt_off /= pnt_wei
        pnt_lm /= pnt_wei

    new_pnt_dict = {'time': time_axes[:, 0],
                    'weight': pnt_wei[:, 0],
                    'direction': pnt_dir,
                    'offset': pnt_off,
                    'lm': pnt_lm,
                    'nsamp': the_shape[0]}
    return new_pnt_dict


def _match_tp_to_time(tp_info, time_axes, meta_dict):
    """
    Match total power data to the new time axes
    Args:
        tp_info: total power info Xarray
        time_axes: The new time axes
        meta_dict: Metadata dictionary containing flags

    Returns:
    Time matched total power data dictionary
    """
    flag_times = meta_dict['flagged_times']
    iflag = 0
    itime = 0
    the_shape = [time_axes.shape[0]]

    tp_amp = np.zeros(the_shape)
    tp_pha = np.zeros(the_shape)
    tp_ref = np.zeros(the_shape)
    tp_wei = np.zeros(the_shape)

    tp_time_axis = tp_info.time.values
    if tp_info.attrs['amplitude_cal']:
        amp_values = tp_info['CALIBRATED_AMPLITUDE'].values
    else:
        amp_values = tp_info['AMPLITUDE'].values
        
    if tp_info.attrs['phase_cal']:
        pha_values = tp_info['CALIBRATED_PHASE'].values
    else:
        pha_values = tp_info['PHASE'].values

    ref_values = tp_info['REFERENCE'].values

    for itp in range(tp_time_axis.shape[0]):
        tp_time = tp_time_axis[itp]
        if tp_time > flag_times[iflag, 1]:
            if iflag == flag_times.shape[0]-1:
                pass
            else:
                iflag += 1
        
        while tp_time > time_axes[itime, 1]:
            if itime == the_shape[0]-1:
                break
            else:
                itime += 1
        if tp_time < time_axes[itime, 0]:
            continue
        elif flag_times[iflag, 0] <= tp_time <= flag_times[iflag, 1]:
            continue

        tp_amp[itime] += amp_values[itp]
        tp_pha[itime] += pha_values[itp]
        tp_ref[itime] += ref_values[itp]
        tp_wei[itime] += 1.0
        
    with np.errstate(divide='ignore', invalid='ignore'):    
        tp_amp /= tp_wei
        tp_pha /= tp_wei
        tp_ref /= tp_wei

    new_tp_dict = {'time': time_axes[:, 0],
                   'amp': tp_amp,
                   'pha': tp_pha,
                   'ref': tp_ref,
                   'weight': tp_wei,
                   'nsamp': the_shape[0]}
    return new_tp_dict


def _match_tp_and_pnt(tp_info, pnt_info):
    """
    Match together total power and pointing info and filtering samples with no data
    Args:
        tp_info: time matched total power dictionary
        pnt_info: time matched pointing dictionary

    Returns:
    Complete dictionary of pointing and total power data
    """
    tp_sel = tp_info['weight'] > 0
    pnt_sel = pnt_info['weight'] > 0
    sel = np.logical_and(tp_sel, pnt_sel)
    
    full_dict = {'time': tp_info['time'][sel],
                 'amp': tp_info['amp'][sel],
                 'pha': tp_info['pha'][sel],
                 'ref': tp_info['ref'][sel],
                 'weight': tp_info['weight'][sel],
                 'direction': pnt_info['direction'][sel],
                 'lm': pnt_info['lm'][sel],
                 'offset': pnt_info['offset'][sel],
                 'nsamp': np.sum(sel)}
    return full_dict


########################
#   Data exporting   ###
########################

def export_data(asdm_name, combined_data, meta_dict, holog_name, int_time,
                verbose):
    """
    Export data to disk in the .holog.zarr format
    Args:
        asdm_name: The name of the input ASDM file
        combined_data: The time matched pointing ad total power dictionary
        meta_dict: The metadata dictionary
        holog_name: The base name for the output file
        int_time: The used integration time
        verbose: print processing messages?
    Returns:
    .holog.zarr on disk
    """
    xds = _data_to_xds(combined_data, meta_dict)
    
    input_dict = _create_base_attr_dict(asdm_name, holog_name)
    input_dict["time_smoothing_interval"] = int_time

    _create_holog_structure(meta_dict, holog_name, input_dict, xds)

    if verbose:
        print(f'Xarray dataset saved to {holog_name}.holog.zarr:\n')
        print(xds)


def _conjugate_beam_data(combined_data):
    time = combined_data['time']*DAY2SEC
    amp = combined_data['amp']
    pha = combined_data['pha']
    wei = combined_data['weight']
    ref = combined_data['ref']
    lm = combined_data['lm']
    off = combined_data['offset']
    ref_median = np.nanmedian(combined_data['ref'])

    n_time = time.shape[0]
    ou_n_time = 2*n_time
    vis_shape = [ou_n_time, 1, 2]  # nchan = 1, npol =2
    pnt_shape = [ou_n_time, 2]

    ou_vis = np.empty(vis_shape, dtype=np.complex128)
    ou_wei = np.empty(vis_shape)
    ou_time = np.empty([ou_n_time])
    ou_lm = np.empty(pnt_shape)
    ou_off = np.empty(pnt_shape)

    for i_time in range(n_time):
        i_ou_time = 2*i_time
        # Time, weight, reference and pointing are equal in conjugate
        ou_time[i_ou_time:i_ou_time+2] = time[i_time]
        ou_wei[i_ou_time:i_ou_time+2, 0, :] = wei[i_time]
        reference = ref[i_time]/ref_median + 0j
        ou_vis[i_ou_time:i_ou_time + 2, 0, 1] = reference
        ou_lm[i_ou_time] = lm[i_time]
        ou_lm[i_ou_time + 1] = lm[i_time]
        ou_off[i_ou_time] = off[i_time]
        ou_off[i_ou_time + 1] = off[i_time]

        real = amp[i_time] * np.cos(pha[i_time])
        imag = amp[i_time] * np.sin(pha[i_time])
        # Value
        ou_vis[i_ou_time, 0, 0] = real + imag*1j
        # its complex conjugate
        ou_vis[i_ou_time+1, 0, 0] = real - imag*1j

    return ou_time, ou_vis, ou_wei, ou_lm, ou_off


def _beam_no_conjugate(combined_data):
    ref_median = np.nanmedian(combined_data['ref'])

    ou_time = combined_data['time']*DAY2SEC
    real_sig = combined_data['amp'] * np.cos(combined_data['pha'])
    imag_sig = combined_data['amp'] * np.sin(combined_data['pha'])
    vis_shape = [ou_time.shape[0], 1, 2]  # nchan = 1, npol =2

    ou_vis = np.empty(vis_shape, dtype=np.complex128)
    ou_vis[:, 0, 0].real = real_sig
    ou_vis[:, 0, 0].imag = imag_sig
    # R2 is to be divided by its median so that we can gauge power variations
    ou_vis[:, 0, 1].real = combined_data['ref'] / ref_median
    ou_vis[:, 0, 1].imag = 0.0

    ou_wei = np.empty(vis_shape)
    ou_wei[:, 0, 0] = combined_data['weight']
    ou_wei[:, 0, 1] = combined_data['weight']

    return ou_time, ou_vis, ou_wei, combined_data['lm'], combined_data['offset']


def _data_to_xds(combined_data, meta_dict):
    """
    Export the time matched pointing and total power data to a Xarray dataset compatible with the astrohackHologFile
    format
    Args:
        combined_data: The time matched pointing ad total power dictionary
        meta_dict: The metadata dictionary
    Returns:
    Xarray dataset compatible with the astrohackHologFile format
    """

    conjugate = False
    if conjugate:
        time, vis, wei, pnt_lm, pnt_off = _conjugate_beam_data(combined_data)
    else:
        time, vis, wei, pnt_lm, pnt_off = _beam_no_conjugate(combined_data)

    coords = {"time": time,
              "chan": np.array(meta_dict['spw']['frequency']),
              "pol": np.array(['I', 'R2'])}

    xds = xr.Dataset()
    xds = xds.assign_coords(coords)
    xds["VIS"] = xr.DataArray(vis, dims=["time", "chan", "pol"])
    xds["WEIGHT"] = xr.DataArray(wei, dims=["time", "chan", "pol"])

    xds["DIRECTIONAL_COSINES"] = xr.DataArray(pnt_lm, dims=["time", "lm"])
    xds["IDEAL_DIRECTIONAL_COSINES"] = xr.DataArray(pnt_off, dims=["time", "lm"])

    # This is not relevant in the NF case, so we leave it at 0
    parallactic_samples = np.array([0, 0, 0])
    extent = _compute_real_extent(combined_data['lm'])

    xds.attrs["holog_map_key"] = "map_0"
    xds.attrs["ddi"] = 0
    xds.attrs["parallactic_samples"] = parallactic_samples
    xds.attrs["telescope_name"] = meta_dict['ant']['telescope']
    xds.attrs["antenna_name"] = meta_dict['ant']['antenna']
    xds.attrs["near_field"] = True
    xds.attrs["nf_focus_off"] = meta_dict['holo']['focus']

    for key, value in extent.items():
        xds.attrs[key] = value
    
    xds.attrs["grid_params"] = _compute_grid_params(meta_dict, extent)
    xds.attrs["time_smoothing_interval"] = combined_data['integ_time']*DAY2SEC

    return xds


def _compute_grid_params(meta_dict, extent):
    """
    Estimate beam gridding parameters
    Code copied but simplified from astrohack
    Args:
        meta_dict: metadata dictionary
        extent: L and M axes extents

    Returns:
    grid parameters
    """
    clight = scipy.constants.speed_of_light
    wavelength = clight / meta_dict['spw']['frequency'][0]

    tel_name = meta_dict['ant']['telescope']+'_'+meta_dict['ant']['antenna'][0:2]
    telescope = Telescope(tel_name.lower())

    cell_size = wavelength/telescope.diam/3.

    min_range = np.min([extent['l_max']-extent['l_min'],
                        extent['m_max']-extent['m_min']])
    n_pix = int(np.ceil(min_range / cell_size)) ** 2
    return {'n_pix': n_pix, 'cell_size': cell_size}


def _compute_real_extent(lm):
    """
    Compute L and M extents
    Args:
        lm: L and M arrays

    Returns:
    Dictionary with L and M extents
    """
    extent = {'l_min': np.min(lm[:, 0]),
              'l_max': np.max(lm[:, 0]),
              'm_min': np.min(lm[:, 1]),
              'm_max': np.max(lm[:, 1])}
    return extent


def _create_holog_structure(meta_dict, holog_name, input_dict, xds):
    """
    Create the .holog.zarr structure in disk
    Args:
        meta_dict: metada dictionary
        holog_name: the base name for the .holg.zarr file
        input_dict: Dictionary with user inputs
        xds: The Xarray dataset with the data on the astrohackHologFile format

    Returns:
    .holog.zarr file on disk
    """
    basename = f'./{holog_name}.holog.zarr'
    import os
    ant_name = 'ant_'+meta_dict['ant']['antenna']
    path = f'{basename}/ddi_0/map_0'
    os.makedirs(path, exist_ok=True)

    # necessary: cell_size, n_pix, telescope_name, that is it!!!
    attr_dict = {"cell_size": xds.attrs['grid_params']["cell_size"],
                 "n_pix": xds.attrs['grid_params']["n_pix"],
                 "telescope_name": meta_dict['ant']['telescope']}
    
    xds_dict = xds.to_dict(data=False)
    holog_dict = {ant_name.lower(): {"ddi_0": {"map_0": xds_dict}}}
    
    _write_dict_as_json(f'./{basename}/.holog_input', input_dict)
    _write_dict_as_json(f'./{basename}/.holog_attr', attr_dict)
    _write_dict_as_json(f'./{basename}/.holog_json', holog_dict, add_origin=False)

    xds.to_zarr(path+'/'+ant_name.lower(), mode='w', compute=True,
                consolidated=True)
    

def _create_base_attr_dict(asdm_name, holog_name):
    """
    Create a basic astrohack inout dictionary
    Args:
        asdm_name: The name of the input ASDM file
        holog_name: The base name of the output .holog.zarr file

    Returns:
    The basic dictionary
    """
    base_dict = {"asdm_name": asdm_name,
                 "point_name": None,
                 "holog_name": holog_name,
                 "holog_obs_dict": None,
                 "ddi": "all",
                 "baseline_average_distance": "all",
                 "baseline_average_nearest": "all",
                 "data_column": None,
                 "parallel": None,
                 "overwrite": None}
    return base_dict


def _write_dict_as_json(file_name, input_dict, add_origin=True):
    """
    Write dictionary as a json file for use in astrohack
    copied from astrohack
    Args:
        file_name: Name for the outpur json file
        input_dict: Dictionary to be written to disk
        add_origin: Add origin and version to dictionary?

    Returns:
    json file in disk
    """
    import json
    import copy

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()

            elif isinstance(obj, np.floating):
                return float(obj)

            elif isinstance(obj, np.integer):
                return int(obj)

            elif isinstance(obj, type(None)):
                return "None"

            return json.JSONEncoder.default(self, obj)
    
    """
    Creates a metadata dictionary that is compatible with JSON and writes it to a file
    Args:
        file_name: Output json file name
        input_dict: Dictionary to be included in the metadata
    """
    meta_data = copy.deepcopy(input_dict)

    if add_origin:
        meta_data.update({
            'version': VERSION,
            'origin': 'ALMA NF filler'
        })

    try:
        with open(file_name, "w") as json_file:
            json.dump(meta_data, json_file, cls=NumpyEncoder)

    except Exception as error:
        print(error)
                

#######################
#   ASDM Printing   ###
#######################
    
def _print_heading(text, wide=60, n_lb=1, sep='*'):
    """
    Print an ASDM table heading
    Args:
        text: The table name
        wide: Separator length
        n_lb: Number of skipped lines after heading
        sep: Separator character to use

    Returns:
    Printed Heading on terminal
    """
    print(wide*sep)
    print(text+n_lb*'\n')

    
def _print_table(table, heading, nmax=4):
    """
    Print an ASDM table to the terminal
    Args:
        table: The ASDM table
        heading: The ASDM table name
        nmax: Maximum number of rows before abbreviating table

    Returns:
    Printed table on terminal
    """
    le_table = table()
    table_rows = le_table.get()
    size = len(table_rows)
    if size == 0:
        _print_heading(heading+' Table is empty')
    elif size > nmax:
        _print_heading(heading+' Table:')
        print(f'Table contains {size} rows, example row (0-eth):')
        print(table_rows[0])
    else:
        _print_heading(heading+' Table:')
        for item in table_rows:
            try:
                print(item)
            except UnicodeEncodeError:
                print('ERROR: Table contains non unicode characters')

                
def print_asdm_summary(asdm_name):
    """
    Print a summary of ASDM tables to the terminal
    Args:
        asdm_name: The ASDM file on disk

    Returns:
    printed ASDM summary on terminal
    """
    asdm_object = _open_asdm(asdm_name)
    print(asdm_object)
    for table in asdm_object.tables():
        _print_table(getattr(asdm_object, table), table)
        
    return
