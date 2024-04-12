from asdm import *
import numpy as np
import xarray as xr
import scipy
from astrohack.antenna.telescope import Telescope
from matplotlib import pyplot as plt
from scipy import interpolate
import time as timer


DAY2SEC = 86400
VERSION = '0.0.1'

################################
###   Direct ASDM handling   ###
################################

def asdm_to_holog(asdm_name, holog_name, int_time, verbose, fake_corr,
                  cal_amp, cal_pha, cal_cycle, plot_cal, save_cal):
    start = timer.time()
    print(f'Processing {asdm_name} to {holog_name}.holog.zarr...\n')
    
    asdm_object = _open_asdm(asdm_name)
    meta_dict = get_asdm_metadata(asdm_object)
    pnt_info = get_pnt_from_asdm(asdm_object, verbose)
    tp_info = get_tp_from_asdm(asdm_object, meta_dict['holo'], verbose)

    calibrated_data = calibrate_tp_data(tp_info, cal_amp, cal_pha, cal_cycle,
                                        plot_cal, holog_name, verbose)
    if save_cal:
        calibrated_data.to_zarr(holog_name+'.cal.zarr', mode='w', compute=True,
                consolidated=True)

    combined_data = combine_data(meta_dict, pnt_info, calibrated_data, int_time,
                                 verbose)
    export_data(asdm_name, combined_data, meta_dict, holog_name, int_time,
                verbose, fake_corr)

    el_time = timer.time()-start
    print(f'\nFinished processing ({el_time:.2f} seconds)')
    

def _open_asdm(asdm_name):
    parser = ASDMParseOptions()
    parser.asALMA()
    parser.loadTablesOnDemand(True)

    asdm_object = ASDM()
    asdm_object.setFromFile(asdm_name,parser)
    return asdm_object


#########################
### Retrieve metadata ###
#########################

def get_asdm_metadata(asdm_object):
    meta_dict = {'holo': _get_holography_info(asdm_object),
                 'spw': _get_frequency_info(asdm_object),
                 'ant': _get_antenna_info(asdm_object),
                 'flagged_times': _get_flag_info(asdm_object)}
    return meta_dict


def _get_frequency_info(asdm_object):
    spw_info = asdm_object.spectralWindowTable().get()[0]
    
    spw_dict = {'nchan': spw_info.numChan(),
                'frequency': np.array([spw_info.refFreq().get()])}
    return spw_dict


def _get_antenna_info(asdm_object):
    ant_info = asdm_object.antennaTable().get()[0]

    tel_dict = {'antenna': ant_info.name(),
                'telescope': 'ALMA'}

    return tel_dict


def _get_holography_info(asdm_object):
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
    flag_table = asdm_object.flagTable().get()
    nflags = len(flag_table)

    flag_times = np.ndarray([nflags, 2])
    for i_row, flag_row in enumerate(flag_table):
        flag_times[i_row, 0] = flag_row.startTime().getMJD()
        flag_times[i_row, 1] = flag_row.endTime().getMJD()

    return flag_times


#############################
###   Retrieve pointing   ###
#############################

def get_pnt_from_asdm(asdm_object, verbose):
    pnt_table = asdm_object.pointingTable().get()
    num_samples = 0
    pnt_time = np.ndarray((0))
    pnt_scan = np.ndarray((0))
    pnt_dir = np.ndarray((0,2))
    pnt_enc = np.ndarray((0,2))
    pnt_tgt = np.ndarray((0,2))
    pnt_off = np.ndarray((0,2))
    
    for irow, pnt_row in enumerate(pnt_table):
        row_time, row_scan, row_dir, row_enc, row_tgt, row_off, n_samp =  _get_pnt_from_row(pnt_row, irow)
        num_samples += n_samp
        
        pnt_time = np.concatenate([pnt_time, row_time])
        pnt_scan = np.concatenate([pnt_scan, row_scan])
        pnt_dir = np.concatenate([pnt_dir, row_dir])
        pnt_enc = np.concatenate([pnt_enc, row_enc])
        pnt_tgt = np.concatenate([pnt_tgt, row_tgt])
        pnt_off = np.concatenate([pnt_off, row_off])

    
    if verbose:    
        print("Total number of pointing samples: ",num_samples)
    pnt_dict = {'time': pnt_time,
                'scan': pnt_scan,
                'direction': pnt_dir,
                'encoder': pnt_enc,
                'target': pnt_tgt,
                'offset': pnt_off,           
    }
    l = np.cos(pnt_dict['target'][:, 1]) * np.sin(pnt_dict['target'][:, 0] - pnt_dict['direction'][:, 0])
    m = np.sin(pnt_dict['target'][:, 1]) * np.cos(pnt_dict['direction'][:, 1]) - np.cos(pnt_dict['target'][:,1])\
        * np.sin(pnt_dict['direction'][:, 1]) * np.cos(pnt_dict['target'][:, 0] - pnt_dict['direction'][:, 0])

    pnt_dict['l'] = l
    pnt_dict['m'] = m
     
    pnt_dict = _filter_axes(pnt_dict, tol=0.0001, ref_axis='l')
    pnt_dict = _filter_axes(pnt_dict, tol=0.0001, ref_axis='m')

    pnt_dict['nsamp'] = pnt_dict['time'].shape[0]

    if verbose:
        print("Total number of valid pointing samples: ", pnt_dict['nsamp'],'\n')
    
    return pnt_dict


def _filter_axes(pnt_dict, tol=0.0001, ref_axis='direction'):
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


def _print_pnt_dict(pnt_dict):
    print('\n############################################################\n')
    
    RAD2DEG = 180/np.pi
    print('X direction, min max', np.min(pnt_dict['direction'][:,0])*RAD2DEG,
          np.max(pnt_dict['direction'][:,0])*RAD2DEG)
    print('Y direction, min max', np.min(pnt_dict['direction'][:,1])*RAD2DEG,
          np.max(pnt_dict['direction'][:,1])*RAD2DEG)
    print('X Offset, min max:', np.min(pnt_dict['offset'][:,0])*RAD2DEG, np.max(pnt_dict['offset'][:,0])*RAD2DEG)
    print('Y Offset, min max:', np.min(pnt_dict['offset'][:,1])*RAD2DEG, np.max(pnt_dict['offset'][:,1])*RAD2DEG)

    l = np.cos(pnt_dict['target'][:, 1]) * np.sin(pnt_dict['target'][:, 0] - pnt_dict['direction'][:, 0])
    m = np.sin(pnt_dict['target'][:, 1]) * np.cos(pnt_dict['direction'][:, 1]) - np.cos(pnt_dict['target'][:,1])\
        * np.sin(pnt_dict['direction'][:, 1]) * np.cos(pnt_dict['target'][:, 0] - pnt_dict['direction'][:, 0])

    print('L, min max', np.min(l)*RAD2DEG, np.max(l)*RAD2DEG)
    print('M, min max', np.min(m)*RAD2DEG, np.max(m)*RAD2DEG)


def _get_pnt_from_row(pnt_row, irow):
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
    # I assume a 2D tuple here
    n_long = len(letuple)
    n_wide = len(letuple[0])
    np_data = np.ndarray((n_long, n_wide))
    for i_long in range(n_long):
        for i_wide in range(n_wide):
            np_data[i_long, i_wide] = letuple[i_long][i_wide].get()
    return np_data


############################
###   Retrieve tp data   ###
############################

def get_tp_from_asdm(asdm_object, holo_info, verbose):
    if verbose:
        print('Retrieving Total Power Table...')
        start = timer.time()
    tp_table = asdm_object.totalPowerTable().get()
    if verbose:
        el_time = timer.time()-start
        print(f'Done retrieving Total Power Table, took {el_time:.2f} seconds')
    num_samples = len(tp_table)
    n_corr = holo_info['ncorr']
    
    tp_time = np.ndarray((num_samples))
    tp_scan = np.ndarray((num_samples))
    tp_r2 = np.ndarray((num_samples))
    tp_qr = np.ndarray((num_samples))
    tp_rs = np.ndarray((num_samples))
    
    idict = _get_corr_indexes(holo_info, ['R2', 'QR', 'RS'])

    for i_row, tp_row in enumerate(tp_table):
        tp_time[i_row] = tp_row.time().getMJD()
        tp_scan[i_row] = tp_row.subscanNumber()
        tp_r2[i_row] = tp_row.floatData()[0][0][idict['R2']]
        tp_qr[i_row] = tp_row.floatData()[0][0][idict['QR']]
        tp_rs[i_row] = tp_row.floatData()[0][0][idict['RS']]

    if verbose:
        print("Total number of Total power samples: ",num_samples,'\n')

    # Real and imag parts from correlations
    tp_s = tp_rs/tp_r2
    tp_q = tp_qr/tp_r2

    # Amplitude = sqrt(Q2 + S2)
    tp_amp = np.sqrt(tp_s**2+tp_q**2)
    
    # Phase = arctan(Q/S)
    tp_pha = np.arctan(tp_q/tp_s)

    tp_dict = {'time': tp_time,
               'scan': tp_scan,
               'ref':  tp_r2,
               'amp':  tp_amp,
               'pha':  tp_pha,
               'nsamp': num_samples}
    return tp_dict


def _get_corr_indexes(holo_info, wanted_corr):
    idict = {}
    for corr in wanted_corr:
        if corr in holo_info['corr_axis']:
            idict[corr] = list(holo_info['corr_axis']).index(corr)
        else:
            Exception(f'ERROR: {corr} correlation not present in data')
    return idict


#######################    
###   Calibration   ###
#######################

def calibrate_tp_data(tp_info, cal_amp, cal_pha, cal_cycle, plot_cal,
                      holog_name, verbose):
    """Calibration scheme derived from:
    Near-Field Radio Holography of Large Reﬂector Antennas
    DOI: 10.1109/MAP.2007.4395293 · Source: IEEE Xplore
    """

    cal_data, data = _separate_cal_from_data(tp_info, cal_cycle)

    _amplitude_calibration(cal_data, data, cal_amp, plot_cal, holog_name, verbose)
    _phase_calibration(cal_data, data, cal_pha, plot_cal, holog_name, verbose)
    if verbose:
        print()
            
    return data


def _separate_cal_from_data(tp_info, cal_cycle):
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
    if cal_type == 'none':
        return None
    elif cal_type in cal_methods.keys():
        return cal_methods[cal_type](cal_time, cal_values, data_time)
    else:
        raise Exception(f"Unknown calibration solving algorithm {cal_fit}")

    
def _spline_fit(cal_time, cal_data, data_time):
    spl_coeff = interpolate.splrep(cal_time, cal_data)
    spl_val = interpolate.splev(data_time, spl_coeff)
    return spl_val


def _mean_fit(cal_time, cal_data, data_time):
    mean_val = np.nanmean(cal_data)
    return np.full_like(data_time, mean_val)


def _linear_regression_fit(cal_time, cal_data, data_time):
    system = np.empty([cal_time.shape[0], 2])
    system[:, 0] = cal_time
    system[:, 1] = 1.0
    
    regression = np.linalg.lstsq(system, cal_data, rcond=None)
    vec = regression[0]
    solution = vec[0]*data_time + vec[1]
    return solution


def _linear_interpolation_fit(cal_time, cal_data, data_time):
    func = interpolate.interp1d(cal_time, cal_data)
    return func(data_time)


def _square_interpolation_fit(cal_time, cal_data, data_time):
    func = interpolate.interp1d(cal_time, cal_data, kind='quadratic')
    return func(data_time)
    

cal_methods = {'linterp': _linear_interpolation_fit,
               'spline': _spline_fit,
               'mean': _mean_fit,
               'regression': _linear_regression_fit,
               'sqinterp': _square_interpolation_fit,
               }

CALIBRATION_OPTIONS = list(cal_methods.keys())
CALIBRATION_OPTIONS.append('none')

def _amplitude_calibration(cal, data, cal_type, plot_cal, holog_name, verbose):
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
        RAD2DEG = 180/np.pi
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


#######################
###   Data mixing   ###
#######################

def combine_data(meta_dict, pnt_info, tp_info, int_time, verbose):
    """This function combines the pointing data with the tp data while
    taking into account the flags, all work here assumes time is
    monotonically crescent.
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
        
        while pnt_time > time_axes[itime,1]:
            if itime == the_shape[0]-1:
                break
            else:
                itime += 1
        if pnt_time < time_axes[itime,0]:
            continue
        elif flag_times[iflag, 0] <= pnt_time <= flag_times[iflag,1]:
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
    tp_sel = tp_info['weight'] > 0
    pnt_sel = pnt_info['weight'] > 0
    sel = np.logical_and(tp_sel, pnt_sel)
    
    full_dict = {'time': tp_info['time'][sel],
                 'amp' : tp_info['amp'][sel],
                 'pha' : tp_info['pha'][sel],
                 'ref' : tp_info['ref'][sel],
                 'weight': tp_info['weight'][sel],
                 'direction': pnt_info['direction'][sel],
                 'lm': pnt_info['lm'][sel],
                 'offset': pnt_info['offset'][sel],
                 'nsamp': np.sum(sel)}
    return full_dict


##########################
###   Data exporting   ###
##########################

def export_data(asdm_name, combined_data, meta_dict, holog_name, int_time,
                verbose, fake_corr):
    xds = _data_to_xds(combined_data, meta_dict, fake_corr)
    
    input_dict = _create_base_attr_dict(asdm_name, holog_name)
    input_dict["time_smoothing_interval"] = int_time

    path = _create_holog_structure(meta_dict, holog_name, input_dict, xds)

    if verbose:
        print(f'Xarray dataset saved to {holog_name}.holog.zarr:\n')
        print(xds)


def _data_to_xds(combined_data, meta_dict, fake_corr):

    # This is not relevant in the NF case so we leave it at 0
    parallactic_samples = np.array([0, 0, 0]) 
    
    extent = _compute_real_extent(combined_data['lm'])

    if fake_corr:
        pol_axis = np.array(['I', 'R2', 'R3', 'R4'])
    else:
        pol_axis = np.array(['I', 'R2'])
              
    coords = {"time": np.array(combined_data['time']*DAY2SEC),
              "chan": np.array(meta_dict['spw']['frequency']),
              "pol": pol_axis}

    vis_shape = [coords['time'].shape[0],
                 coords['chan'].shape[0],
                 coords['pol'].shape[0]]

    real_sig = combined_data['amp'] * np.cos(combined_data['pha'])
    imag_sig = combined_data['amp'] * np.sin(combined_data['pha'])

    vis = np.empty(vis_shape, dtype=np.complex128)         
    vis[:,0,0].real = real_sig
    vis[:,0,0].imag = imag_sig
    vis[:,0,1].real = combined_data['ref']
    vis[:,0,1].imag = 0.0
    if fake_corr:
        vis[:,0,2].real = combined_data['ref']
        vis[:,0,2].imag = 0.0
        vis[:,0,3].real = combined_data['ref']
        vis[:,0,3].imag = 0.0

    wei = np.empty(vis_shape)
    wei[:, 0, 0] = combined_data['weight']
    wei[:, 0, 1] = combined_data['weight']
    
    xds = xr.Dataset()
    xds = xds.assign_coords(coords)
    xds["VIS"] = xr.DataArray(vis, dims=["time", "chan", "pol"])
    xds["WEIGHT"] = xr.DataArray(wei, dims=["time", "chan", "pol"])

    xds["DIRECTIONAL_COSINES"] = \
        xr.DataArray(combined_data['lm'], dims=["time", "lm"])
    xds["IDEAL_DIRECTIONAL_COSINES"] = \
        xr.DataArray(combined_data['offset'], dims=["time", "lm"])

    xds.attrs["holog_map_key"] = "map_0"
    xds.attrs["ddi"] = 0
    xds.attrs["parallactic_samples"] = parallactic_samples
    xds.attrs["telescope_name"] = meta_dict['ant']['telescope']
    xds.attrs["antenna_name"] = meta_dict['ant']['antenna']
    xds.attrs["near_field"] = True

    for key, value in extent.items():
        xds.attrs[key] = value
    
    xds.attrs["grid_params"] = _compute_grid_params(meta_dict, extent)
    xds.attrs["time_smoothing_interval"] = combined_data['integ_time']*DAY2SEC

    return xds


def _compute_grid_params(meta_dict, extent):
    # Code copied but simplified from astrohack
    clight = scipy.constants.speed_of_light
    wavelength = clight / meta_dict['spw']['frequency'][0]

    tel_name = meta_dict['ant']['telescope']+'_'+meta_dict['ant']['antenna'][0:2]
    telescope = Telescope(tel_name)

    cell_size = 0.85 * wavelength/telescope.diam

    min_range = np.min([extent['l_max']-extent['l_min'],
                        extent['m_max']-extent['m_min']])
    n_pix = int(np.ceil(min_range / cell_size)) ** 2
    return {'n_pix': n_pix, 'cell_size': cell_size}


def _compute_real_extent(lm):
    extent = {'l_min': np.min(lm[:, 0]),
              'l_max': np.max(lm[:, 0]),
              'm_min': np.min(lm[:, 1]),
              'm_max': np.max(lm[:, 1])}
    return extent


def _create_holog_structure(meta_dict, holog_name, input_dict, xds):
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
                

#########################
###   ASDM Printing   ###
#########################
    
def _print_heading(text, wide=60, n_lb=1, sep='*'):
    print(wide*sep)
    print(text+n_lb*'\n')

    
def _print_table(table, heading, nmax = 4):
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
    excluded_methods = ['entity', 'setFromFile']
    asdm_object = _open_asdm(asdm_name)
    print(asdm_object)
    for table in asdm_object.tables():
        _print_table(getattr(asdm_object, table), table)
        
    return




