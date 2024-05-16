import time
import numpy as np
from graphviper.utils import logger as logger
from numba import njit
from numba.core import types
from matplotlib import pyplot as plt

from astrohack.utils import calc_coords


def convolution_gridding(grid_type, visibilities, weights, lmvis, diameter, freq, sky_cell_size, grid_size):

    beam_size = _compute_beam_size(diameter, freq)
    if grid_type == 'exponential':
        pass
    else:
        msg = f'Unknown grid type {grid_type}.'
        logger.error(msg)
        raise Exception(msg)

    nchan = visibilities.shape[1]
    if nchan > 1:
        msg = 'Convolution gridding only supported for a single channel currently'
        logger.error(msg)
        raise Exception(msg)

    logger.info('Creating convolved beam...')
    start = time.time()
    laxis, maxis = calc_coords(grid_size, sky_cell_size)
    beam, wei = _convolution_gridding_jit(visibilities, lmvis, weights, sky_cell_size, laxis, maxis, beam_size)
    duration = time.time()-start
    logger.info(f'Convolution took {duration:.3} seconds')
    return beam


@njit(cache=False, nogil=True)
def _convolution_gridding_jit(visibilities, lmvis, weights, sky_cell_size, laxis, maxis, beam_size):
    ntime, nchan, npol = visibilities.shape

    l_kernel = _create_exponential_kernel(beam_size, sky_cell_size[0])
    m_kernel = _create_exponential_kernel(beam_size, sky_cell_size[1])

    grid_shape = (nchan, npol, laxis.shape[0], maxis.shape[0])
    beam_grid = np.zeros(grid_shape, dtype=types.complex128)
    weig_grid = np.zeros(grid_shape)

    for i_time in range(ntime):

        lval, mval = lmvis[i_time]
        i_lmin, i_lmax = _compute_kernel_range(l_kernel, lval, laxis)
        i_mmin, i_mmax = _compute_kernel_range(m_kernel, mval, maxis)

        for i_chan in range(nchan):
            for i_pol in range(npol):
                for il in range(i_lmin, i_lmax):
                    l_fac = _convolution_factor(l_kernel, laxis[il] - lval)
                    for im in range(i_mmin, i_mmax):
                        m_fac = _convolution_factor(m_kernel, maxis[im] - mval)

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
def _create_exponential_kernel(beam_size, cell_size, exponent=2):
    smoothing = beam_size
    support = 4*smoothing
    width = smoothing/(2*np.sqrt(np.log(2.0)))

    pix_support = support/np.abs(cell_size)
    pix_width = width/np.abs(cell_size)
    if pix_support < 1.0:
        used_support = 2*(pix_support + 0.995) + 1
    else:
        used_support = 2*pix_support + 1

    kernel_size = used_support * 100 + 1
    bias = 50.0 * used_support + 1.0
    u_axis = (np.arange(kernel_size) - bias) * 0.01

    ker_dict = {'bias': bias,
                'u-axis': u_axis,
                'kernel': np.exp(-(u_axis/pix_width)**exponent),
                'user_support': support,
                'user_width': width,
                'pix_support': pix_support,
                'cell_size': cell_size}
    return ker_dict


@njit(cache=False, nogil=True)
def _compute_kernel_range(kernel, coor, axis):
    idx = _find_nearest(coor, axis)
    i_min = idx - kernel['pix_support']
    i_max = idx + kernel['pix_support']

    if i_min < 0:
        i_min = 0
    if i_max >= axis.shape[0]:
        i_max = axis.shape[0] - 1
    return i_min, i_max


@njit(cache=False, nogil=True)
def _convolution_factor(kernel, delta):
    pix_delta = delta/np.abs(kernel['cell_size'])
    ikern = round(100.0*pix_delta+kernel['bias'])
    return kernel['kernel'][ikern]


@njit(cache=False, nogil=True)
def _compute_kernel_correction(kernel, grid_size):
    correction = np.zeros(grid_size)
    ker_val = kernel['kernel']
    bias = kernel['bias']
    m_point = grid_size/2 + 1

    kw_coeff = 0.01 * np.pi / m_point
    kernel_extent = 2*bias+1
    for i_kern in range(kernel_extent):
        if ker_val[i_kern] != 0:
            kx_coeff  = kw_coeff*(i_kern-bias)
            for i_corr in range(grid_size):
                correction[i_corr] += ker_val[i_kern] * np.cos(kx_coeff*(i_corr-m_point))

    return correction


def _compute_beam_size(diameter, frequency):
    if isinstance(frequency, np.ndarray):
        freq = frequency[0]
    else:
        freq = frequency
    # This beam size is anchored at NOEMA beam measurements we might need a more general formula
    size = 41 * (115e9 / freq) * np.sqrt(2.) * (15. / diameter) * np.pi / 180 / 3600
    return size


def gridding_correction(aperture, freq, diameter, sky_cell_size, u_axis, v_axis):
    beam_size = _compute_beam_size(diameter, freq)

    return _gridding_correction_jit(aperture, beam_size, sky_cell_size, u_axis, v_axis)


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







