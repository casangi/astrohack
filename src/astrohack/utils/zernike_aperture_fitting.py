import numpy as np
from scipy import optimize as opt

from astrohack.utils import create_2d_array_reconstruction_array
from astrohack.utils.algorithms import create_aperture_mask

# Cartesian forms for the Zernike Polynomials extracted from Lakshminarayanan & Fleck, Journal of modern Optics 2011.


def zernike_order_0(u_ax, v_ax):
    """
    Zernike 0-eth order polynomial, simple flat disk.
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis

    Returns:
        a [n, 1] Matrix filled with ones.
    """
    # N = 0
    return np.full([u_ax.shape[0], 1], 1.0)


def zernike_order_1(u_ax, v_ax):
    """
    Zernike first order polynomials, simple linear gradients.
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis

    Returns:
        a [n, 3] Matrix filled with U and V values.
    """
    # N = 1
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 3])
    matrix[:, 0] = 1.0
    # M = -1
    matrix[:, 1] = u_ax
    # M = 1
    matrix[:, 2] = v_ax
    return matrix


def zernike_order_2(u_ax, v_ax, return_powers=False):
    """
    Zernike second order polynomials, first astigmatisms and linear defocus.
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis
        return_powers: return list of already compute U and V powers? (saves computing time)

    Returns:
        a [n, 6] Matrix filled with the first 6 polynomials.
    """
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 6])
    # Fill with previous order
    matrix[:, 0:3] = zernike_order_1(u_ax, v_ax)
    u_pow = [None, None, u_ax**2]
    v_pow = [None, None, v_ax**2]

    # M = -2
    matrix[:, 3] = 2 * u_ax * v_ax
    # M = 0
    matrix[:, 4] = -1 + 2 * u_pow[2] + 2 * v_pow[2] ** 2
    # M = 2
    matrix[:, 5] = -u_pow[2] ** 2 + v_pow[2] ** 2

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_3(u_ax, v_ax, return_powers=False):
    """
    Zernike third order polynomials, first comas and trefoils.
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis
        return_powers: return list of already compute U and V powers? (saves computing time)

    Returns:
        a [n, 10] Matrix filled the first 10 polynomials.
    """
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 10])
    # Fill with previous order
    matrix[:, 0:6], u_pow, v_pow = zernike_order_2(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[2] * u_ax)
    v_pow.append(v_pow[2] * v_ax)

    # M = -3
    matrix[:, 6] = -u_pow[3] + 3 * u_ax * v_pow[2]
    # M = -1
    matrix[:, 7] = -2 * u_ax + 3 * u_pow[3] + 3 * u_ax * v_pow[2]
    # M = 1
    matrix[:, 8] = -2 * v_ax + 3 * v_pow[3] + 3 * u_pow[2] * v_ax
    # M = 3
    matrix[:, 9] = -v_pow[3] + 3 * v_pow[3] + 3 * u_pow[2] * v_ax

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_4(u_ax, v_ax, return_powers=False):
    """
    Zernike forth order polynomials, second astigmatisms, first spherical aberations and tetrafoils.
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis
        return_powers: return list of already compute U and V powers? (saves computing time)

    Returns:
        a [n, 15] Matrix filled the first 15 polynomials.
    """
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 15])
    # Fill with previous order
    matrix[:, 0:10], u_pow, v_pow = zernike_order_3(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[2] ** 2)
    v_pow.append(v_pow[2] ** 2)

    # M = -4
    matrix[:, 10] = -4 * u_pow[3] * v_ax + 4 * u_ax * v_pow[3]
    # M = -2
    matrix[:, 11] = -6 * u_ax * v_ax + 8 * u_pow[3] * v_ax + 8 * u_ax * v_pow[3]
    # M = 0
    matrix[:, 12] = (
        1
        - 6 * u_pow[2]
        - 6 * v_pow[2]
        + 6 * u_pow[4]
        + 12 * u_pow[2] * v_pow[2]
        + 6 * v_pow[4]
    )
    # M = 2
    matrix[:, 13] = 3 * u_pow[2] - 3 * v_pow[2] - 4 * u_pow[4] + 4 * v_pow[4]
    # M = 4
    matrix[:, 14] = u_pow[4] - 6 * u_pow[2] * v_pow[2] + v_pow[4]

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_5(u_ax, v_ax, return_powers=False):
    """
    Zernike fifth order polynomials, pentafoils, second comas.
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis
        return_powers: return list of already compute U and V powers? (saves computing time)

    Returns:
        a [n, 21] Matrix filled the first 21 polynomials.
    """
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 21])
    # Fill with previous order
    matrix[:, 0:15], u_pow, v_pow = zernike_order_4(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[4] * u_ax)
    v_pow.append(v_pow[4] * v_ax)

    # M = -5
    matrix[:, 15] = u_pow[5] - 10 * u_pow[3] * v_pow[2] + 5 * u_ax * v_pow[4]
    # M = -3
    matrix[:, 16] = (
        4 * u_pow[3]
        - 12 * u_ax * v_pow[2]
        - 5 * u_pow[5]
        + 10 * u_pow[3] * v_pow[2]
        + 15 * u_ax * v_pow[4]
    )
    # M = -1
    matrix[:, 17] = (
        3 * u_ax
        - 12 * u_pow[3]
        - 12 * u_ax * v_pow[2]
        + 10 * u_pow[5]
        + 20 * u_pow[3] * v_pow[2]
        + 10 * u_ax * v_pow[4]
    )
    # M = 1
    matrix[:, 18] = (
        3 * v_ax
        - 12 * v_pow[3]
        - 12 * v_ax * u_pow[2]
        + 10 * v_pow[5]
        + 20 * v_pow[3] * u_pow[2]
        + 10 * v_ax * u_pow[4]
    )
    # M = 3
    matrix[:, 19] = (
        -4 * v_pow[3]
        + 12 * v_ax * u_pow[2]
        + 5 * v_pow[5]
        - 10 * v_pow[3] * u_pow[2]
        - 15 * v_ax * u_pow[4]
    )
    # M = 5
    matrix[:, 20] = v_pow[5] - 10 * v_pow[3] * u_pow[2] + 5 * v_ax * u_pow[4]

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_6(u_ax, v_ax, return_powers=False):
    """
    Zernike sixth order polynomials, second spherical aberations, hexafoils
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis
        return_powers: return list of already compute U and V powers? (saves computing time)

    Returns:
        a [n, 28] Matrix filled the first 28 polynomials.
    """
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 28])
    # Fill with previous order
    matrix[:, 0:21], u_pow, v_pow = zernike_order_5(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[5] * u_ax)
    v_pow.append(v_pow[5] * v_ax)
    u3v = u_pow[3] * v_ax
    uv3 = u_ax * v_pow[3]
    u5v = u_pow[5] * v_ax
    uv5 = u_ax * v_pow[5]
    u3v3 = u_pow[3] * v_pow[3]
    u2v4 = u_pow[2] * v_pow[4]
    u4v2 = u_pow[4] * v_pow[2]
    u2v2 = u_pow[2] * v_pow[2]

    # M = -6
    matrix[:, 21] = 6 * u5v - 20 * u3v3 + 6 * uv5
    # M = -4
    matrix[:, 22] = 20 * u3v - 20 * uv3 - 24 * u5v + 24 * uv5
    # M = -2
    matrix[:, 23] = (
        12 * u_ax * v_ax - 40 * u3v - 40 * uv3 + 30 * u5v + 60 * u3v3 - 30 * uv5
    )
    # M = 0
    matrix[:, 24] = (
        -1
        + 12 * u_pow[2]
        + 12 * v_pow[2]
        - 30 * u_pow[4]
        - 60 * u2v2
        - 30 * v_pow[4]
        + 20 * u_pow[6]
        + 60 * u4v2
        + 60 * u2v4
        + 20 * v_pow[6]
    )
    # M = 2
    matrix[:, 25] = (
        -6 * u_pow[2]
        + 6 * v_pow[2]
        + 20 * u_pow[4]
        - 20 * v_pow[4]
        - 15 * u_pow[6]
        - 15 * u4v2
        + 15 * u2v4
        + 15 * v_pow[6]
    )
    # M = 4
    matrix[:, 26] = (
        -5 * u_pow[4]
        + 30 * u2v2
        - 5 * v_pow[4]
        + 6 * u_pow[6]
        - 30 * u4v2
        - 30 * u2v4
        + 6 * v_pow[6]
    )
    # M = 6
    matrix[:, 27] = -u_pow[6] + 15 * u4v2 - 15 * u2v4 + v_pow[6]

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_7(u_ax, v_ax, return_powers=False):
    """
    Zernike seventh order polynomials, heptafoils
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis
        return_powers: return list of already compute U and V powers? (saves computing time)

    Returns:
        a [n, 36] Matrix filled the first 36 polynomials.
    """
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 36])
    # Fill with previous order
    matrix[:, 0:28], u_pow, v_pow = zernike_order_6(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[6] * u_ax)
    v_pow.append(v_pow[6] * v_ax)

    u5v2 = u_pow[5] * v_pow[2]
    u2v5 = u_pow[2] * v_pow[5]
    u3v4 = u_pow[3] * v_pow[4]
    u4v3 = u_pow[4] * v_pow[3]
    u3v2 = u_pow[3] * v_pow[2]
    u2v3 = u_pow[2] * v_pow[3]
    uv2 = u_ax * v_pow[2]
    u2v = u_pow[2] * v_ax
    uv4 = u_ax * v_pow[4]
    u4v = u_pow[4] * v_ax
    uv6 = u_ax * v_pow[6]
    u6v = u_pow[6] * v_ax

    # M = -7
    matrix[:, 28] = -u_pow[7] + 21 * u5v2 - 35 * u3v4 + 7 * uv6
    # M = -5
    matrix[:, 29] = (
        -6 * u_pow[5]
        + 60 * u3v2
        - 30 * uv4
        + 7 * u_pow[7]
        - 63 * u5v2
        - 35 * u3v4
        + 35 * uv6
    )
    # M = -3
    matrix[:, 30] = (
        -10 * u_pow[3]
        + 30 * uv2
        + 30 * u_pow[5]
        - 60 * u3v2
        - 90 * uv4
        - 21 * u_pow[7]
        + 21 * u5v2
        + 105 * u3v4
        + 63 * uv6
    )
    # M = -1
    matrix[:, 31] = (
        -4 * u_ax
        + 30 * u_pow[3]
        + 30 * uv2
        - 60 * u_pow[5]
        - 120 * u3v2
        - 60 * uv4
        - 35 * u_pow[7]
        + 105 * u5v2
        + 105 * u3v4
        + 35 * uv6
    )
    # M = 1
    matrix[:, 32] = (
        -4 * v_ax
        + 30 * v_pow[3]
        + 30 * u2v
        - 60 * v_pow[5]
        - 120 * u2v3
        - 60 * u4v
        - 35 * v_pow[7]
        + 105 * u2v5
        + 105 * u4v3
        + 35 * u6v
    )
    # M = 3 -> Symmetry would suggest that there is a typo here on the paper hence the u6v term having a negative
    # signal here but not in Lakshminarayanan & Fleck 2011
    matrix[:, 33] = (
        10 * v_pow[3]
        - 30 * u2v
        - 30 * v_pow[5]
        + 60 * u2v3
        + 90 * u4v
        + 21 * v_pow[7]
        - 21 * u2v5
        - 105 * u4v3
        + 63 * u6v
    )
    # M = 5
    matrix[:, 34] = (
        -6 * v_pow[5]
        + 60 * u2v3
        - 30 * u4v
        + 7 * v_pow[7]
        - 63 * u2v5
        - 35 * u4v3
        + 35 * u6v
    )
    # M = 7
    matrix[:, 35] = v_pow[7] - 21 * u2v5 + 35 * u4v3 - 7 * u6v

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_8(u_ax, v_ax, return_powers=False):
    """
    Zernike eigth order polynomials, octafoils
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis
        return_powers: return list of already compute U and V powers? (saves computing time)

    Returns:
        a [n, 45] Matrix filled the first 45 polynomials.
    """
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 45])
    # Fill with previous order
    matrix[:, 0:36], u_pow, v_pow = zernike_order_7(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[7] * u_ax)
    v_pow.append(v_pow[7] * v_ax)

    u2v2 = u_pow[2] * v_pow[2]
    u3v3 = u_pow[3] * v_pow[3]
    u4v4 = u_pow[4] * v_pow[4]

    u3v = u_pow[3] * v_ax
    uv3 = u_ax * v_pow[3]
    u5v = u_pow[5] * v_ax
    uv5 = u_ax * v_pow[5]
    u7v = u_pow[7] * v_ax
    uv7 = u_ax * v_pow[7]
    u5v3 = u_pow[5] * v_pow[3]
    u3v5 = u_pow[3] * v_pow[5]
    u2v4 = u_pow[2] * v_pow[4]
    u4v2 = u_pow[4] * v_pow[2]
    u2v6 = u_pow[2] * v_pow[6]
    u6v2 = u_pow[6] * v_pow[2]

    # M -8
    matrix[:, 36] = 8 * (uv7 - u7v) + 56 * (u5v3 - u3v5)
    # M -6
    matrix[:, 37] = (
        -42 * (u5v + uv5) + 140 * u3v3 + 48 * (u7v + uv7) - 112 * (u5v3 + u3v5)
    )
    # M -4
    matrix[:, 38] = (
        60 * (uv3 - u3v) + 168 * (u5v - uv5) + 112 * (uv7 - u7v) + 112 * (u3v5 - u5v3)
    )
    # M -2
    matrix[:, 39] = (
        -20 * u_ax * v_ax
        + 120 * (u3v + uv3)
        - 210 * (u5v + uv5)
        - 420 * u3v3
        + 112 * (uv7 - u7v)
        + 336 * (u5v3 + u3v5)
    )
    # M 0
    matrix[:, 40] = (
        1
        - 20 * (u_pow[2] + v_pow[2])
        + 90 * (u_pow[4] + v_pow[4])
        + 180 * u2v2
        - 140 * (u_pow[6] + v_pow[6])
        - 420 * (u4v2 + u2v4)
        + 70 * (u_pow[8] + v_pow[8])
        + 280 * (u2v6 + u6v2)
        + 420 * u4v4
    )
    # M 2
    matrix[:, 41] = (
        10 * (u_pow[2] - v_pow[2])
        - 60 * (v_pow[4] - u_pow[4])
        + 105 * (u4v2 - u2v4)
        + 105 * (u_pow[6] - v_pow[6])
        + 56 * (v_pow[8] - u_pow[8])
        + 112 * (u2v6 - u6v2)
    )
    # M 4
    matrix[:, 42] = (
        15 * (u_pow[4] + v_pow[4])
        - 90 * u2v2
        - 42 * (u_pow[6] + v_pow[6])
        + 210 * (u2v4 + u4v2)
        + 28 * (u_pow[8] + v_pow[8])
        - 112 * (u2v6 + u6v2)
        - 280 * u4v4
    )
    # M 6
    matrix[:, 43] = (
        7 * (u_pow[6] - v_pow[6])
        + 105 * (u2v4 - u4v2)
        + 8 * (v_pow[8] - u_pow[8])
        + 112 * (u6v2 - u2v6)
    )
    # M 8
    matrix[:, 44] = u_pow[8] + v_pow[8] - 28 * (u6v2 + u2v6) + 70 * u4v4

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_9(u_ax, v_ax, return_powers=False):
    """
    Zernike nineth order polynomials, enneafoils
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis
        return_powers: return list of already compute U and V powers? (saves computing time)

    Returns:
        a [n, 55] Matrix filled the first 55 polynomials.
    """
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 55])
    # Fill with previous order
    matrix[:, 0:45], u_pow, v_pow = zernike_order_8(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[8] * u_ax)
    v_pow.append(v_pow[8] * v_ax)

    uv8 = u_ax * v_pow[8]
    u8v = u_pow[8] * v_ax
    uv6 = u_ax * v_pow[6]
    u6v = u_pow[6] * v_ax
    uv4 = u_ax * v_pow[4]
    u4v = u_pow[4] * v_ax
    uv2 = u_ax * v_pow[2]
    u2v = u_pow[2] * v_ax

    u7v2 = u_pow[7] * v_pow[2]
    u2v7 = u_pow[2] * v_pow[7]
    u5v4 = u_pow[5] * v_pow[4]
    u4v5 = u_pow[4] * v_pow[5]
    u3v6 = u_pow[3] * v_pow[6]
    u6v3 = u_pow[6] * v_pow[3]
    u5v2 = u_pow[5] * v_pow[2]
    u2v5 = u_pow[2] * v_pow[5]
    u3v4 = u_pow[3] * v_pow[4]
    u4v3 = u_pow[4] * v_pow[3]
    u3v2 = u_pow[3] * v_pow[2]
    u2v3 = u_pow[2] * v_pow[3]

    # M = -9
    matrix[:, 45] = u_pow[9] - 36 * u7v2 + 126 * u5v4 - 84 * u3v6 + 9 * uv8
    # M = -7
    matrix[:, 46] = (
        8 * u_pow[7]
        - 168 * u5v2
        + 280 * u3v4
        - 56 * uv6
        - 9 * u_pow[9]
        + 180 * u7v2
        - 126 * u5v4
        - 252 * u3v6
        + 63 * uv8
    )
    # M = -5
    matrix[:, 47] = (
        21 * u_pow[5]
        - 210 * u3v2
        + 105 * uv4
        - 56 * u_pow[7]
        + 504 * u5v2
        + 280 * u3v4
        - 280 * uv6
        + 36 * u_pow[9]
        - 288 * u7v2
        - 504 * u5v4
        + 180 * uv8
    )
    # M = -3
    matrix[:, 48] = (
        20 * u_pow[3]
        - 60 * uv2
        - 105 * u_pow[5]
        + 210 * u3v2
        + 315 * uv4
        + 168 * u_pow[7]
        - 168 * u5v2
        - 840 * u3v4
        - 504 * uv6
        - 84 * u_pow[9]
        + 504 * u5v4
        + 672 * u3v6
        + 252 * uv8
    )
    # M = -1
    matrix[:, 49] = (
        5 * u_ax
        - 60 * u_pow[3]
        - 60 * uv2
        + 210 * u_pow[5]
        + 420 * u3v2
        + 210 * uv4
        - 280 * u_pow[7]
        - 840 * u5v2
        - 840 * u3v4
        - 280 * uv6
        + 126 * u_pow[9]
        + 504 * u7v2
        + 756 * u5v4
        + 128 * uv8
    )
    # M = 1
    matrix[:, 50] = (
        5 * v_ax
        - 60 * v_pow[3]
        - 60 * u2v
        + 210 * v_pow[5]
        + 420 * u2v3
        + 210 * u4v
        - 280 * v_pow[7]
        - 840 * u2v5
        - 840 * u4v3
        - 280 * u6v
        + 126 * v_pow[9]
        + 504 * u2v7
        + 756 * u4v5
        + 128 * u8v
    )
    # M = 3
    matrix[:, 51] = (
        -20 * v_pow[3]
        + 60 * u2v
        + 105 * v_pow[5]
        - 210 * u2v3
        - 315 * u4v
        - 168 * v_pow[7]
        + 168 * u2v5
        + 840 * u4v3
        + 504 * u6v
        + 84 * v_pow[9]
        - 504 * u4v5
        - 672 * u6v3
        - 252 * u8v
    )
    # M = 5
    matrix[:, 52] = (
        21 * v_pow[5]
        - 210 * u2v3
        + 105 * u4v
        - 56 * v_pow[7]
        + 504 * u2v5
        + 280 * u4v3
        - 280 * u6v
        + 36 * v_pow[9]
        - 288 * u2v7
        - 504 * u4v5
        + 180 * u8v
    )
    # M = 7
    matrix[:, 53] = (
        -8 * v_pow[7]
        + 168 * u2v5
        - 280 * u4v3
        + 56 * u6v
        + 9 * v_pow[9]
        - 180 * u2v7
        + 126 * u4v5
        + 252 * u6v3
        - 63 * u8v
    )
    # M = 9
    matrix[:, 54] = v_pow[9] - 36 * u2v7 + 126 * u4v5 - 84 * u6v3 + 9 * u8v

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_10(u_ax, v_ax, return_powers=False):
    """
    Zernike tenth order polynomials, decafoils
    Args:
        u_ax: Aperture U axis
        v_ax: Aperture V axis
        return_powers: return list of already compute U and V powers? (saves computing time)

    Returns:
        a [n, 66] Matrix filled the first 66 polynomials.
    """
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 66])
    # Fill with previous order
    matrix[:, 0:55], u_pow, v_pow = zernike_order_9(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[9] * u_ax)
    v_pow.append(v_pow[9] * v_ax)

    u2v2 = u_pow[2] * v_pow[2]
    u3v3 = u_pow[3] * v_pow[3]
    u4v4 = u_pow[4] * v_pow[4]
    u5v5 = u_pow[5] * v_pow[5]

    u3v = u_pow[3] * v_ax
    uv3 = u_ax * v_pow[3]
    u5v = u_pow[5] * v_ax
    uv5 = u_ax * v_pow[5]
    u7v = u_pow[7] * v_ax
    uv7 = u_ax * v_pow[7]
    uv9 = u_ax * v_pow[9]
    u9v = u_pow[9] * v_ax
    u7v3 = u_pow[7] * v_pow[3]
    u3v7 = u_pow[3] * v_pow[7]
    u5v3 = u_pow[5] * v_pow[3]
    u3v5 = u_pow[3] * v_pow[5]
    u2v4 = u_pow[2] * v_pow[4]
    u4v2 = u_pow[4] * v_pow[2]
    u2v6 = u_pow[2] * v_pow[6]
    u6v2 = u_pow[6] * v_pow[2]
    u2v8 = u_pow[2] * v_pow[8]
    u8v2 = u_pow[8] * v_pow[2]
    u4v6 = u_pow[4] * v_pow[6]
    u6v4 = u_pow[6] * v_pow[4]

    # M = -10
    matrix[:, 55] = 10 * (u9v + uv9) - 120 * (u7v3 + u3v7) + 252 * u5v5
    # M = -8
    matrix[:, 56] = (
        72 * (u7v - uv7) + 504 * (u3v5 - u5v3) + 80 * (uv9 - u9v) + 480 * (u7v3 - u3v7)
    )
    # M = -6
    matrix[:, 57] = (
        270 * (u9v + uv9)
        - 360 * (u7v3 + u3v7)
        - 1260 * u5v5
        - 432 * (u7v + uv7)
        + 1008 * (u5v3 + u3v5)
        + 168 * (u5v + uv5)
        - 560 * u3v3
    )
    # M = -4
    matrix[:, 58] = (
        140 * (u3v - uv3)
        + 672 * (uv5 - u5v)
        + 1008 * (u7v - uv7)
        + 1008 * (u5v3 - u3v5)
        + 480 * (uv9 - u9v)
        + 960 * (u3v7 - u7v3)
    )
    # M = -2
    matrix[:, 59] = (
        30 * u_ax * v_ax
        - 280 * (u3v + uv3)
        + 840 * (u5v + uv5)
        + 1680 * u3v3
        - 1008 * (u7v + uv7)
        - 3024 * (u3v5 + u5v3)
        + 420 * (u9v + uv9)
        + 1680 * (u7v3 + u3v7)
        + 2520 * u5v5
    )
    # M = 0
    matrix[:, 60] = (
        -1
        + 30 * (u_pow[2] + v_pow[2])
        - 210 * (u_pow[4] + v_pow[4])
        - 420 * u2v2
        + 560 * (u_pow[6] + v_pow[6])
        + 1680 * (u2v4 + u4v2)
        - 630 * (u_pow[8] + v_pow[8])
        - 2520 * (u6v2 + u2v6)
        - 3780 * u4v4
        + 252 * (u_pow[10] + v_pow[10])
        + 1260 * (u8v2 + u2v8)
    ) + 2520 * (u6v4 + u4v6)
    # M = 2
    matrix[:, 61] = (
        15 * (v_pow[2] - u_pow[2])
        + 140 * (u_pow[4] - v_pow[4])
        + 420 * (v_pow[6] - u_pow[6])
        + 420 * (u2v4 - u4v2)
        + 504 * (u_pow[8] - v_pow[8])
        + 1008 * (u6v2 - u2v6)
        + 210 * (v_pow[10] - u_pow[10])
        + 630 * (u2v8 - u8v2)
        + 420 * (u4v6 - u6v4)
    )
    # M = 4
    matrix[:, 62] = (
        -35 * (u_pow[4] + v_pow[4])
        + 210 * u2v2
        + 168 * (u_pow[6] + v_pow[6])
        - 840 * (u4v2 + u2v4)
        - 252 * (u_pow[8] + v_pow[8])
        + 1008 * (u6v2 + u2v6)
        + 2520 * u4v4
        + 120 * (u_pow[10] + v_pow[10])
        - 360 * (u8v2 + u2v8)
        - 1680 * (u6v4 + u4v6)
    )
    # M = 6
    matrix[:, 63] = (
        28 * (v_pow[6] - u_pow[6])
        + 420 * (u4v2 - u2v4)
        + 72 * (u_pow[8] - v_pow[8])
        + 1008 * (u2v6 - u6v2)
        + 45 * (v_pow[10] - u_pow[10])
        + 585 * (u8v2 - u2v8)
        + 630 * (u6v4 - u4v6)
    )
    # M = 8
    matrix[:, 64] = (
        -9 * (u_pow[8] + v_pow[8])
        + 252 * (u6v2 + u2v6)
        - 630 * u4v4
        + 10 * (u_pow[10] + v_pow[10])
        - 270 * (u8v2 + u2v8)
        + 420 * (u6v4 + u4v6)
    )
    # M = 10
    matrix[:, 65] = (v_pow[10] - u_pow[10]) + 45 * (u8v2 - u2v8) + 210 * (u4v6 - u6v4)

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


zernike_matrix_functions = [
    zernike_order_0,
    zernike_order_1,
    zernike_order_2,
    zernike_order_3,
    zernike_order_4,
    zernike_order_5,
    zernike_order_6,
    zernike_order_7,
    zernike_order_8,
    zernike_order_9,
    zernike_order_10,
]


def create_osa_coordinates(zernike_order, split_nm=False):
    """
    Create a list of the labels of all the polynomials from the N order given.
    Args:
        zernike_order: The value of the N order.
        split_nm: Is result to be given in a 1D list (False, default) or in a 2D [N,M] list.

    Returns:
        A list containing the OSA ordered Zernike indices labels.
    """
    coordinates = []
    for n in range(zernike_order + 1):
        if n == 0:
            coordinates.append([0, 0])
        else:
            for m in range(-n, n + 1, 2):
                coordinates.append([n, m])

    if not split_nm:
        coordinates = [f"N={idx[0]}, M={idx[1]}" for idx in coordinates]

    return coordinates


def fit_zernike_coefficients(
    aperture,
    u_axis,
    v_axis,
    zernike_order,
    telescope,
    fitting_engine="numpy",
    mask_arm_shadows=True,
):
    """
    Fit zernike polynomial coefficients to an aperture array.
    Args:
        aperture: 5D array containing the correlation apertures ([time, chan, pol, u, v]).
        u_axis: U axis for the apertures
        v_axis: V axis for the apertures.
        zernike_order: The N-eth order to be fitted to the aperture.
        telescope: Telescope class object with telescope's optical info.
        fitting_engine: Which fittign engine to use, numpy's linear algebra or scipy optimize.
        mask_arm_shadows: Mask the shadows of the arms holding the secondary in place (usually only relevant for
        cassegrain telescopes)

    Returns:
        The Zernike coefficients, the model derived from these coefficients, the RMS of the fit and the labeling for
        the zernike coeficients.
    """
    # Creating a unitary radius grid
    aperture_radius = telescope.diam / 2
    u_grid, v_grid = np.meshgrid(u_axis, v_axis, indexing="ij")
    u_grid /= aperture_radius
    v_grid /= aperture_radius

    # Creating a mask for valid points
    if mask_arm_shadows:
        mask = create_aperture_mask(
            u_axis,
            v_axis,
            telescope.inlim,
            aperture_radius,
            arm_width=telescope.arm_shadow_width,
            arm_angle=telescope.arm_shadow_rotation,
        )
    else:
        mask = create_aperture_mask(u_axis, v_axis, telescope.inlim, aperture_radius)

    # Vectorize grids with only valid points
    u_lin = u_grid[mask]
    v_lin = v_grid[mask]
    aperture_4d = aperture[:, :, :, mask]

    # Creating grid reconstruction
    uv_idx_grid = create_2d_array_reconstruction_array(u_axis, v_axis, mask)

    matrix_func = zernike_matrix_functions[zernike_order]
    matrix = matrix_func(u_lin, v_lin)

    ntime, nchan, npol = aperture.shape[:3]
    ncoeffs = matrix.shape[1]
    zernike_coeffs = np.ndarray([ntime, nchan, npol, ncoeffs], dtype=complex)
    zernike_model = np.full_like(aperture, np.nan + np.nan * 1j, dtype=complex)
    fit_rms = np.ndarray([ntime, nchan, npol], dtype=complex)

    if fitting_engine == "numpy":
        fit_func = _fit_an_aperture_plane_component_np_least_squares
    elif fitting_engine == "scipy":
        fit_func = _fit_an_aperture_plane_component_scipy_opt_lst_sq
    else:
        raise Exception(f"Unknown fitting method {fitting_engine}")

    for itime in range(ntime):
        for ichan in range(nchan):
            for ipol in range(npol):
                real_coeffs, real_rms, real_model = fit_func(
                    matrix, aperture_4d[itime, ichan, ipol].real
                )
                imag_coeffs, imag_rms, imag_model = fit_func(
                    matrix, aperture_4d[itime, ichan, ipol].imag
                )
                zernike_model[
                    itime, ichan, ipol, uv_idx_grid[:, 0], uv_idx_grid[:, 1]
                ] = (real_model[:] + 1j * imag_model[:])
                zernike_coeffs[itime, ichan, ipol] = real_coeffs + 1j * imag_coeffs
                fit_rms[itime, ichan, ipol] = real_rms + 1j * imag_rms

    return zernike_coeffs, zernike_model, fit_rms, create_osa_coordinates(zernike_order)


def _fit_an_aperture_plane_component_np_least_squares(matrix, aperture_plane_comp):
    """
    Fit Zernike polynomial coefficients to aperture data using numpy's linear algebra least squares engine.
    Args:
        matrix: The matrix containing the Zernike polynomials
        aperture_plane_comp: a real or imaginary part of an aperture plane (usually a correlation)

    Returns:
    coefficients, fit rms, and model
    """
    max_ap = np.nanmax(aperture_plane_comp)
    result, _, _, _ = np.linalg.lstsq(matrix, aperture_plane_comp / max_ap, rcond=None)
    model = max_ap * np.matmul(matrix, result)
    rms = np.sqrt(np.sum((aperture_plane_comp - model) ** 2)) / model.shape[0]
    return result, rms, model


def _scipy_fitting_func(coeffs, matrix, aperture):
    """
    Just a simple minimizing function to be used with scipy optimize least squares engine.
    Args:
        coeffs: Coefficients to be tested.
        matrix: The matrix containing the Zernike polynomials.
        aperture: Aperture data.

    Returns:
        residuals of aperture - model
    """
    return aperture - np.matmul(matrix, coeffs)


def _fit_an_aperture_plane_component_scipy_opt_lst_sq(matrix, aperture_plane_comp):
    """
    Fit Zernike polynomial coefficients to aperture data using scipy's optimize least squares engine.
    Args:
        matrix: The matrix containing the Zernike polynomials
        aperture_plane_comp: a real or imaginary part of an aperture plane (usually a correlation)

    Returns:
        coefficients, fit rms, and model
    """
    initial_pars = np.ones(matrix.shape[1])
    max_ap = np.nanmax(aperture_plane_comp)
    norm_aperture = aperture_plane_comp / max_ap
    args = [matrix, norm_aperture]
    results = opt.least_squares(_scipy_fitting_func, initial_pars, args=args)
    model = max_ap * np.matmul(matrix, results.x)
    rms = np.sqrt(np.sum((aperture_plane_comp - model) ** 2)) / model.shape[0]
    return results.x, rms, model
