import time

import numpy as np
from graphviper.utils import logger as logger
from numba import njit
from numba.experimental import jitclass
from numba.core import types
from matplotlib import pyplot as plt

from astrohack.utils import calc_coords


@njit(cache=False, nogil=True)
def find_nearest(value, array):
    diff = np.abs(array-value)
    idx = diff.argmin()
    #print(diff)
    # print(idx, value, array[idx], diff[idx])
    return idx
    # idx = np.searchsorted(array, value, side="left")
    # if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
    #     return idx-1
    # else:
    #     return idx


class_interface = [('bias', types.float64),
                   ('u_axis', types.float64[:]),
                   ('kernel', types.float64[:]),
                   ('user_support', types.float64),
                   ('user_width', types.float64),
                   ('pix_support', types.float64),
                   ('cell_size', types.float64)]


@jitclass(class_interface)
class ExponentialConvolutionKernel:
    def __init__(self, support, width, exponent, cell_size):
        pix_support = support/np.abs(cell_size)
        pix_width = width/np.abs(cell_size)
        if pix_support < 1.0:
            used_support = 2*(pix_support + 0.995) + 1
        else:
            used_support = 2*pix_support + 1
        kernel_size = used_support * 100 + 1
        self.bias = 50.0 * used_support + 1.0
        self.u_axis = (np.arange(kernel_size)-self.bias)*0.01
        self.kernel = np.exp(-(self.u_axis/pix_width)**exponent)
        self.user_support = support
        self.user_width = width
        self.pix_support = pix_support
        self.cell_size = cell_size

    def factor(self, delta):
        pix_delta = delta/np.abs(self.cell_size)
        ikern = round(100.0*pix_delta+self.bias)
        #print(delta, pix_delta, ikern, self.kernel.shape, self.kernel[ikern])
        return self.kernel[ikern]

    def compute_range(self, coor, axis):
        idx = find_nearest(coor, axis)
        i_min = idx - self.pix_support
        i_max = idx + self.pix_support

        if i_min < 0:
            i_min = 0
        if i_max >= axis.shape[0]:
            i_max = axis.shape[0] - 1
        return i_min, i_max

    def plot(self):
        plt.plot(self.u_axis, self.kernel)
        plt.show()
        plt.clf()




def exponential_convolution_kernel(support, width, exponent):
    # Number of rows
    used_support = int(max(support + 0.995, 1.0))
    print(used_support, width)
    used_support = used_support * 2 + 1
    kernel_size = used_support * 100 + 1


    bias = 50.0 * used_support + 1.0

    u_axis = np.abs(np.arange(kernel_size)-bias*0.01)
    u_axis = (np.arange(kernel_size)-bias)*0.01
    kernel = np.exp(-(u_axis/width)**exponent)
    #kernel = np.where(u_axis > support, 0, np.exp(-(u_axis/width)**exponent))

    ker_dict = {'kernel': kernel,
                'bias': bias,
                'support': support}

    # xaxis = np.arange(kernel_size)
    # plt.plot(u_axis, kernel)
    # plt.show()

    return ker_dict


def convolution_gridding(grid_type, visibilities, weights, lmaxis, diameter, freq, sky_cell_size, grid_size):
    # This beam size is anchored at NOEMA beam measurements we might need a more general formula
    beam_size = 41*(115e9/freq[0])*np.sqrt(2.)*(15./diameter)*np.pi/180/3600
    #smoothing = beam_size/2.0
    smoothing = beam_size
    support = 4*smoothing
    width = smoothing/(2*np.sqrt(np.log(2.0)))
    # l_support = support/np.abs(sky_cell_size[0])
    # m_support = support/np.abs(sky_cell_size[1])
    # l_width = smoothing/(2*np.sqrt(np.log(2.0)))/np.abs(sky_cell_size[0])
    # m_width = smoothing/(2*np.sqrt(np.log(2.0)))/np.abs(sky_cell_size[1])
    # print(support, l_width)
    exponent = 2

    if grid_type == 'exponential':
        # print('l', l_support, l_width, exponent)
        # l_kernel = exponential_convolution_kernel(l_support, l_width, exponent)
        # print('m', m_support, m_width, exponent)
        # m_kernel = exponential_convolution_kernel(m_support, m_width, exponent)
        l_kernel = ExponentialConvolutionKernel(support, width, exponent, sky_cell_size[0])
        m_kernel = ExponentialConvolutionKernel(support, width, exponent, sky_cell_size[1])
        #l_kernel.plot()

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
    beam, wei = convolution_gridding_jit(visibilities, lmaxis, weights, l_kernel, m_kernel, sky_cell_size, grid_size,
                                         laxis, maxis)
    duration = time.time()-start
    logger.info(f'Convolution took {duration:.3} seconds')
    return beam


@njit(cache=False, nogil=True)
def convolution_gridding_jit(visibilities, lmaxis, weights, l_kernel, m_kernel, sky_cell_size, grid_size, laxis, maxis):
    ntime, nchan, npol = visibilities.shape

    grid_shape = (nchan, npol, laxis.shape[0], maxis.shape[0])
    beam_grid = np.zeros(grid_shape, dtype=types.complex128)
    weig_grid = np.zeros(grid_shape)

    for i_time in range(ntime):
        # if i_time % 100 == 0:
        #     print(f'{100*i_time/ntime:.1f}% done')

        lval, mval = lmaxis[i_time]
        # #print('l')
        # i_lmin, i_lmax = compute_i_min_max(lval, laxis, l_kernel['support'], sky_cell_size[0], grid_size[0])
        # #print('m')
        # i_mmin, i_mmax = compute_i_min_max(mval, maxis, m_kernel['support'], sky_cell_size[1], grid_size[1])
        i_lmin, i_lmax = l_kernel.compute_range(lval, laxis)
        i_mmin, i_mmax = m_kernel.compute_range(mval, maxis)

        for i_chan in range(nchan):
            for i_pol in range(npol):
                for il in range(i_lmin, i_lmax):
                    # dl = (laxis[il]-lval)/sky_cell_size[0]
                    dl = laxis[il] - lval
                    for im in range(i_mmin, i_mmax):
                        # dm = (maxis[im]-mval)/np.abs(sky_cell_size[1])
                        dm = maxis[im]-mval

                        # print('l', laxis[il], lval, sky_cell_size[0], dl)
                        # print('m', maxis[im], mval, sky_cell_size[1], dm)
                        # conv_fact = (conv_factor(dl, l_kernel) * conv_factor(dm, m_kernel) *
                        #              weights[i_time, i_chan, i_pol])
                        l_fac = l_kernel.factor(dl)
                        m_fac = m_kernel.factor(dm)
                        # print(dl, l_fac)
                        # print(dm, m_fac)

                        conv_fact = l_fac * m_fac * weights[i_time, i_chan, i_pol]

                        beam_grid[i_chan, i_pol, il, im] += conv_fact*visibilities[i_time, i_chan, i_pol]
                        weig_grid[i_chan, i_pol, il, im] += conv_fact

    # plt.imshow(np.abs(beam_grid[0, 0, ...]))
    # plt.show()

    beam_grid /= weig_grid
    return beam_grid, weig_grid





@njit(cache=False, nogil=True)
def compute_i_min_max(coor, axis, support, cell_size, grid_size):
    idx = find_nearest(coor, axis)
    pix_support = np.ceil(np.abs(support/cell_size))[0]
    i_min = idx - support
    i_max = idx + support

    if i_min < 0:
        i_min = 0
    if i_max >= grid_size:
        i_max = grid_size - 1

    # print(idx, type(idx))
    # print(support)
    # print(pix_support, type(pix_support))
    # print(i_min, type(i_min))
    # print(i_max, type(i_max))

    return int(i_min), int(i_max)


@njit(cache=False, nogil=True)
def conv_factor(delta, kernel):
    # print(delta)
    ikern = round(100.0*delta+kernel['bias'])
    value = kernel['kernel'][ikern]
    # print(value)
    return value
