import time
import numpy as np
from graphviper.utils import logger as logger
from scipy.interpolate import griddata
from numba import njit
from numba.core import types
from matplotlib import pyplot as plt

from astrohack.utils import sig_2_fwhm, find_nearest, calc_coords, find_peak_beam_value, chunked_average
from astrohack.utils.text import get_str_idx_in_list


def grid_beam(ant_ddi_dict, grid_size, sky_cell_size, avg_chan, chan_tol_fac, telescope, grid_interpolation_mode):
    n_holog_map = len(ant_ddi_dict.keys())
    map0 = list(ant_ddi_dict.keys())[0]
    freq_axis = ant_ddi_dict[map0].chan.values
    pol_axis = ant_ddi_dict[map0].pol.values
    n_chan = ant_ddi_dict[map0].sizes["chan"]
    n_pol = ant_ddi_dict[map0].sizes["pol"]
    # Get near field status

    reference_scaling_frequency = np.mean(freq_axis)
    if avg_chan:
        avg_chan_map, avg_freq = _create_average_chan_map(freq_axis, chan_tol_fac)
        n_chan = 1
        freq_axis = [np.mean(avg_freq)]
    else:
        avg_chan_map = None
        avg_freq = None

    l_axis, m_axis, l_grid, m_grid, beam_grid = _create_beam_grid(grid_size, sky_cell_size, n_chan, n_pol, n_holog_map)
    scipy_interp = ['linear', 'nearest', 'cubic']

    time_centroid = []
    for holog_map_index, holog_map in enumerate(ant_ddi_dict.keys()):
        ant_xds = ant_ddi_dict[holog_map]

        # Grid the data
        vis = ant_xds.VIS.values
        vis[vis == np.nan] = 0.0
        lm = ant_xds.DIRECTIONAL_COSINES.values
        weight = ant_xds.WEIGHT.values

        if grid_interpolation_mode in scipy_interp:
            grid_corr = False
            beam_grid[holog_map_index, ...] = _scipy_gridding(vis, weight, lm, l_grid, m_grid, grid_interpolation_mode,
                                                              avg_chan_map, avg_freq, reference_scaling_frequency)
        elif grid_interpolation_mode == 'gaussian':
            grid_corr = True
            beam_grid[holog_map_index, ...] = _convolution_gridding(vis, weight, lm, telescope.diam, freq_axis, l_axis,
                                                                    m_axis, sky_cell_size)
        else:
            msg = f'Unknown grid type {grid_interpolation_mode}.'
            logger.error(msg)
            raise Exception(msg)

        time_centroid_index = ant_ddi_dict[holog_map].sizes["time"] // 2
        time_centroid.append(ant_ddi_dict[holog_map].coords["time"][time_centroid_index].values)

        beam_grid[holog_map_index, ...] = _normalize_beam(beam_grid[holog_map_index, ...], n_chan, pol_axis)

    return beam_grid, time_centroid, freq_axis, pol_axis, l_axis, m_axis, grid_corr


def gridding_correction(aperture, freq, diameter, sky_cell_size, u_axis, v_axis):
    beam_size = _compute_beam_size(diameter, freq)
    return _gridding_correction_jit(aperture, beam_size, sky_cell_size, u_axis, v_axis)


def _create_beam_grid(grid_size, sky_cell_size, n_chan, n_pol, n_map):

    l_axis, m_axis = calc_coords(grid_size, sky_cell_size)
    l_grid, m_grid = list(map(np.transpose, np.meshgrid(l_axis, m_axis)))

    beam_grid = np.zeros((n_map,) + (n_chan, n_pol) + l_grid.shape, dtype=np.complex128)

    return l_axis, m_axis, l_grid, m_grid, beam_grid


def _scipy_gridding(vis, weight, lm, l_grid, m_grid, grid_interpolation_mode, avg_chan_map, avg_freq,
                    reference_scaling_frequency):
    logger.info('Interpolating beam onto grid...')
    start = time.time()
    if avg_freq is not None:
        vis_avg, weight_sum = chunked_average(vis, weight, avg_chan_map, avg_freq)
        lm_freq_scaled = lm[:, :, None] * (avg_freq / reference_scaling_frequency)
        n_pol = vis_avg.shape[2]
        beam_grid = np.zeros((1, n_pol, l_grid.shape[0], l_grid.shape[1]), dtype=complex)
        # Unavoidable for loop because lm change over frequency.
        for i_chan in range(avg_freq.shape[0]):
            # Average scaled beams.
            gridded_chan = np.moveaxis(griddata(lm_freq_scaled[:, :, i_chan], vis_avg[:, i_chan, :], (l_grid, m_grid),
                                                method=grid_interpolation_mode, fill_value=0.0), 2, 0)
            beam_grid[0, :, :, :] += gridded_chan
    else:
        beam_grid = np.moveaxis(griddata(lm, vis, (l_grid, m_grid), method=grid_interpolation_mode, fill_value=0.0),
                                (0, 1), (2, 3))
    duration = time.time()-start
    logger.info(f'Interpolation took {duration:.3} seconds')
    return beam_grid


def _normalize_beam(beam_grid, n_chan, pol_axis):
    if 'I' in pol_axis:
        i_i = get_str_idx_in_list('I', pol_axis)
        i_peak = find_peak_beam_value(beam_grid[0, i_i, ...], scaling=0.25)
        beam_grid[0, i_i, ...] /= i_peak
    else:
        if 'RR' in pol_axis:
            i_p1 = get_str_idx_in_list('RR', pol_axis)
            i_p2 = get_str_idx_in_list('LL', pol_axis)
        elif 'XX' in pol_axis:
            i_p1 = get_str_idx_in_list('XX', pol_axis)
            i_p2 = get_str_idx_in_list('YY', pol_axis)
        else:
            msg = f'Unknown polarization scheme: {pol_axis}'
            logger.error(msg)
            raise Exception(msg)

        for chan in range(n_chan):
            try:
                p1_peak = find_peak_beam_value(beam_grid[chan, i_p1, ...], scaling=0.25)
                p2_peak = find_peak_beam_value(beam_grid[chan, i_p2, ...], scaling=0.25)
            except Exception:
                center_pixel = np.array(beam_grid.shape[-2:]) // 2
                p1_peak = beam_grid[chan, i_p1, center_pixel[0], center_pixel[1]]
                p2_peak = beam_grid[chan, i_p2, center_pixel[0], center_pixel[1]]

            normalization = np.abs(0.5 * (p1_peak + p2_peak))

            if normalization == 0:
                logger.warning("Peak of zero found! Setting normalization to unity.")
                normalization = 1

                beam_grid[chan, ...] /= normalization
    return beam_grid


def _create_average_chan_map(freq_chan, chan_tolerance_factor):
    n_chan = len(freq_chan)

    tol = np.max(freq_chan) * chan_tolerance_factor
    n_pb_chan = int(np.floor((np.max(freq_chan) - np.min(freq_chan)) / tol) + 0.5)

    # Create PB's for each channel
    if n_pb_chan == 0:
        n_pb_chan = 1

    if n_pb_chan >= n_chan:
        cf_chan_map = np.arange(n_chan)
        pb_freq = freq_chan
        return cf_chan_map, pb_freq

    pb_delta_bandwdith = (np.max(freq_chan) - np.min(freq_chan)) / n_pb_chan
    pb_freq = (
            np.arange(n_pb_chan) * pb_delta_bandwdith
            + np.min(freq_chan)
            + pb_delta_bandwdith / 2
    )

    cf_chan_map = np.zeros((n_chan,), dtype=int)
    for i in range(n_chan):
        cf_chan_map[i], _ = find_nearest(pb_freq, freq_chan[i])

    return cf_chan_map, pb_freq


def _convolution_gridding(visibilities, weights, lmvis, diameter, freq, l_axis, m_axis, sky_cell_size):
    beam_size = _compute_beam_size(diameter, freq)
    nchan = visibilities.shape[1]
    if nchan > 1:
        msg = 'Convolution gridding only supported for a single channel currently'
        logger.error(msg)
        raise Exception(msg)

    logger.info('Creating convolved beam...')
    start = time.time()
    beam, wei = _convolution_gridding_jit(visibilities, lmvis, weights, sky_cell_size, l_axis, m_axis, beam_size)
    duration = time.time()-start
    logger.info(f'Convolution took {duration:.3} seconds')
    return beam


@njit(cache=False, nogil=True)
def _convolution_gridding_jit(visibilities, lmvis, weights, sky_cell_size, l_axis, m_axis, beam_size):
    ntime, nchan, npol = visibilities.shape

    l_kernel = _create_exponential_kernel(beam_size, sky_cell_size[0])
    m_kernel = _create_exponential_kernel(beam_size, sky_cell_size[1])

    grid_shape = (nchan, npol, l_axis.shape[0], m_axis.shape[0])
    beam_grid = np.zeros(grid_shape, dtype=types.complex128)
    weig_grid = np.zeros(grid_shape)

    for i_time in range(ntime):

        lval, mval = lmvis[i_time]
        i_lmin, i_lmax = _compute_kernel_range(l_kernel, lval, l_axis)
        i_mmin, i_mmax = _compute_kernel_range(m_kernel, mval, m_axis)

        for i_chan in range(nchan):
            for i_pol in range(npol):
                for il in range(i_lmin, i_lmax):
                    l_fac = _convolution_factor(l_kernel, l_axis[il] - lval)
                    for im in range(i_mmin, i_mmax):
                        m_fac = _convolution_factor(m_kernel, m_axis[im] - mval)
                        conv_fact = l_fac * m_fac * weights[i_time, i_chan, i_pol]
                        beam_grid[i_chan, i_pol, il, im] += conv_fact*visibilities[i_time, i_chan, i_pol]
                        weig_grid[i_chan, i_pol, il, im] += conv_fact

    beam_grid /= weig_grid
    beam_grid = np.nan_to_num(beam_grid)
    return beam_grid, weig_grid


@njit(cache=False, nogil=True)
def _find_nearest(value, array):
    diff = np.abs(array-value)
    idx = diff.argmin()
    return idx


@njit(cache=False, nogil=True)
def _create_exponential_kernel(beam_size, sky_cell_size, exponent=2):
    smoothing = beam_size
    support = 4*smoothing
    width = smoothing/sig_2_fwhm

    pix_support = support/np.abs(sky_cell_size)
    pix_width = width/np.abs(sky_cell_size)
    if pix_support < 1.0:
        used_support = 2*(pix_support + 0.995) + 1
    else:
        used_support = 2*pix_support + 1

    kernel_size = used_support * 100 + 1
    bias = 50.0 * used_support + 1.0
    u_axis = (np.arange(kernel_size) - bias) * 0.01
    kernel = np.exp(-(u_axis/pix_width)**exponent)

    ker_dict = {'bias': bias,
                'u-axis': u_axis,
                'kernel': kernel,
                'user_support': support,
                'user_width': width,
                'pix_support': pix_support,
                'sky_cell_size': sky_cell_size}
    return ker_dict


@njit(cache=False, nogil=True)
def _compute_kernel_range(kernel, coor, axis):
    idx = _find_nearest(coor, axis)
    i_min = round(idx - kernel['pix_support'])
    i_max = round(idx + kernel['pix_support'])

    if i_min < 0:
        i_min = 0
    if i_max >= axis.shape[0]:
        i_max = axis.shape[0] - 1
    return i_min, i_max


@njit(cache=False, nogil=True)
def _convolution_factor(kernel, delta):
    pix_delta = delta/np.abs(kernel['sky_cell_size'])
    ikern = round(100.0*pix_delta+kernel['bias'])
    return kernel['kernel'][ikern]


@njit(cache=False, nogil=True)
def _compute_kernel_correction(kernel, grid_size):
    correction = np.zeros(grid_size)
    ker_val = kernel['kernel']
    bias = kernel['bias']
    m_point = grid_size/2 + 1

    kw_coeff = 0.01 * np.pi / m_point
    for i_kern in range(kernel.shape[0]):
        if ker_val[i_kern] > 1e-30:
            kx_coeff = kw_coeff*(i_kern-bias)
            for i_corr in range(grid_size):
                costerm = np.cos(kx_coeff*(i_corr-m_point))
                correction[i_corr] += ker_val[i_kern] * costerm

    return correction


def _compute_beam_size(diameter, frequency):
    if isinstance(frequency, (np.ndarray, list, tuple)):
        freq = frequency[0]
    else:
        freq = frequency
    # This beam size is anchored at NOEMA beam measurements we might need a more general formula
    size = 41 * (115e9 / freq) * np.sqrt(2.) * (15. / diameter) * np.pi / 180 / 3600
    return size


@njit(cache=False, nogil=True)
def _get_normalized_correction(u_corr, v_corr):
    u_size = u_corr.shape[0]
    v_size = v_corr.shape[0]
    u_mid = int(np.floor(u_size/2)+1)
    v_mid = int(np.floor(v_size/2)+1)
    norm_coeff = u_corr[u_mid]*v_corr[v_mid]
    norm_corr = np.zeros((u_size, v_size), dtype=types.float64)
    for i_u in range(u_size):
        for i_v in range(v_size):
            norm_corr[i_u, i_v] = u_corr[i_u] * v_corr[i_v] / norm_coeff
    return norm_corr


@njit(cache=False, nogil=True)
def _gridding_correction_jit(aperture, beam_size, sky_cell_size, u_axis, v_axis):
    l_kernel = _create_exponential_kernel(beam_size, sky_cell_size[0])
    m_kernel = _create_exponential_kernel(beam_size, sky_cell_size[1])

    u_corr = _compute_kernel_correction(l_kernel, u_axis.shape[0])
    v_corr = _compute_kernel_correction(m_kernel, v_axis.shape[0])
    norm_corr = _get_normalized_correction(u_corr, v_corr)
    ntime, nchan, npol = aperture.shape[:3]
    for i_time in range(ntime):
        for i_chan in range(nchan):
            for i_pol in range(npol):
                aperture[i_time, i_chan, i_pol] /= norm_corr
    return aperture







