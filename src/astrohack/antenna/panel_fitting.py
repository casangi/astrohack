import numpy as np
import scipy.optimize as opt

import graphviper.utils.logger as logger

from astrohack.utils import gauss_elimination, least_squares

###################################
#  General purpose                #
###################################


def build_system(size):
    matrix = np.zeros([size, size])
    vector = np.zeros([size])
    return matrix, vector

###################################
#  Mean                           #
###################################


def solve_mean(_self, samples):
    """
    Fit panel surface as a simple mean of its points Z deviation
    """
    mean = 0
    nsamp = len(samples)
    if nsamp > 0:
        for point in samples:
            mean += point.value
        mean /= nsamp
    return [mean]


def correct_mean(self, point):
    return point.ix, point.iy, self.parameters[0]

###################################
# Rigid                           #
###################################


def solve_rigid(self, samples):
    """
    Fit panel surface using AIPS gaussian elimination model for rigid panels
    """
    npar = 3
    matrix, vector = build_system(self.npar)
    for point in samples:
        matrix[0, 0] += point.xc * point.xc
        matrix[0, 1] += point.xc * point.yc
        matrix[0, 2] += point.xc
        matrix[1, 0] = matrix[0, 1]
        matrix[1, 1] += point.yc * point.yc
        matrix[1, 2] += point.yc
        matrix[2, 0] = matrix[0, 2]
        matrix[2, 1] = matrix[1, 2]
        matrix[2, 2] += 1.0
        vector[0] += point.value * point.xc
        vector[1] += point.value * point.yc
        vector[2] += point.value

    return gauss_elimination(matrix, vector)


def correct_rigid(self, point):
    """
    Computes fitted value for point [xcoor, ycoor] using AIPS gaussian elimination model for rigid panels
    Args:
        xc: X coordinate of point
        yc: Y coordinate of point

    Returns:
    Fitted value at xcoor,ycoor
    """
    corr = point.xc * self.parameters[0] + point.yc * self.parameters[1] + self.parameters[2]
    return point.ix, point.iy, corr

###################################
# Flexible                        #
###################################


def solve_flexible(self, samples):
    # this can only work for ringed panels...
    system, vector = build_system(self.npar)
    for point in samples:
        auno, aduo, atre, aqua = self._flexible_coeffs(point)
        system[0, 0] += auno*auno
        system[0, 1] += auno*aduo
        system[0, 2] += auno*atre
        system[0, 3] += auno*aqua
        system[1, 1] += aduo*aduo
        system[1, 2] += aduo*atre
        system[1, 3] += aduo*aqua
        system[2, 2] += atre*atre
        system[2, 3] += atre*aqua
        system[3, 3] += aqua*aqua
        vector[0]   += point.value*auno
        vector[1]   += point.value*aduo
        vector[2]   += point.value*atre
        vector[3]   += point.value*aqua

    system[1, 0] = system[0, 1]
    system[2, 0] = system[0, 2]
    system[2, 1] = system[1, 2]
    system[3, 0] = system[0, 3]
    system[3, 1] = system[1, 3]
    system[3, 2] = system[2, 3]
    return gauss_elimination(system, vector)


def correct_flexible(self, point):
    coeffs = self._flexible_coeffs(point)
    return point.ix, point.iy, np.sum(coeffs * self.parameters)

###################################
# Full 9 parameters paraboloid    #
###################################


def solve_full_paraboloid(self, samples):
    """
    Builds the designer matrix for least squares fitting, and calls the _least_squares fitter for a fully fledged
    9 parameter paraboloid
    """
    # ax2y2 + bx2y + cxy2 + dx2 + ey2 + gxy + hx + iy + j
    nsamp = len(samples)
    system = np.full((nsamp, self.npar), 1.0)
    vector = np.zeros(nsamp)
    for ipnt, point in enumerate(samples):
        system[ipnt, 0] = point.xc**2 * point.yc**2
        system[ipnt, 1] = point.xc**2 * point.yc
        system[ipnt, 2] = point.yc**2 * point.xc
        system[ipnt, 3] = point.xc ** 2
        system[ipnt, 4] = point.yc ** 2
        system[ipnt, 5] = point.xc * point.yc
        system[ipnt:, 6] = point.xc
        system[ipnt:, 7] = point.yc
        vector[ipnt] = point.value

    params, _, _ = least_squares(system, vector)
    return params


def correct_full_paraboloid(self, point):
    """
    Computes the correction from the fitted parameters to the 9 parameter paraboloid at (xcoor, ycoor)
    Args:
        xcoor: Coordinate of point in X
        ycoor: Coordinate of point in Y
    Returns:
        The correction at point
    """
    # ax2y2 + bx2y + cxy2 + dx2 + ey2 + gxy + hx + iy + j
    xsq = point.xc**2
    ysq = point.yc**2
    correction = self.parameters[0]*xsq*ysq + self.parameters[1]*xsq*point.yc + self.parameters[2]*ysq*point.xc
    correction += self.parameters[3]*xsq + self.parameters[4]*ysq + self.parameters[5]*point.xc*point.yc
    correction += self.parameters[6]*point.xc + self.parameters[7]*point.yc + self.parameters[8]
    return point.ix, point.iy, correction

#######################################
# Co-rotated paraboloid least squares #
#######################################


def solve_corotated_lst_sq(self, samples):
    """
    Builds the designer matrix for least squares fitting, and calls the _least_squares fitter for a corotated
    paraboloid centered at the center of the panel
    """
    # a*u**2 + b*v**2 + c
    nsamp = len(samples)
    system = np.full((nsamp, self.npar), 1.0)
    vector = np.zeros(nsamp)
    x0 = self.center.xc
    y0 = self.center.yc
    for ipnt, point in enumerate(samples):
        system[ipnt, 0] = ((point.xc - x0) * np.cos(self.zeta) - (point.yc - y0) * np.sin(self.zeta))**2  # U
        system[ipnt, 1] = ((point.xc - x0) * np.sin(self.zeta) + (point.yc - y0) * np.cos(self.zeta))**2  # V
        vector[ipnt] = point.value
        
    params, _, _ = least_squares(system, vector)
    return params


def correct_corotated_lst_sq(self, point):
    """
    Computes the correction from the least squares fitted parameters to the corotated paraboloid
    """
    corrval = corotated_paraboloid_scipy([point.xc, point.yc, self.center.xc, self.center.yc, self.zeta],
                                         *self.parameters)
    return point.ix, point.iy, corrval

###################################
# Co-rotated robust               #
###################################


def solve_corotated_robust(self, samples):
    """
    Try fitting the Surface of a panel using the corotated least_squares method, if that fails fallback to scipy
    fitting
    """
    try:
        return solve_corotated_lst_sq(self, samples)
    except np.linalg.LinAlgError:
        return solve_scipy(self, samples)

###################################
# Scipy base                      #
###################################


def solve_scipy(self, samples, verbose=False, x0=None):
    devia = np.ndarray([len(samples)])
    coords = np.ndarray([5, len(samples)])
    for ipoint, point in enumerate(samples):
        devia[ipoint] = point.value
        coords[:, ipoint] = point.xc, point.yc, self.center.xc, self.center.yc, self.zeta

    liminf = [-np.inf, -np.inf, -np.inf]
    limsup = [np.inf, np.inf, np.inf]
    if x0 is None:
        p0 = [1e2, 1e2, np.mean(devia)]
    else:
        p0 = x0
    if self.npar > len(p0):
        liminf.append(0.0)
        limsup.append(np.pi)
        p0.append(0)

    maxfevs = [100000, 1000000, 10000000]
    for maxfev in maxfevs:
        try:
            result = opt.curve_fit(self.fitting_function, coords, devia,
                                   p0=p0, bounds=[liminf, limsup], maxfev=maxfev)
        except RuntimeError:
            if verbose:
                logger.info("Increasing number of iterations")
            continue
        else:
            params = result[0]
            if verbose:
                logger.info("Converged with less than {0:d} iterations".format(maxfev))
            return params


def correct_scipy(self, point):
    corrval = self.fitting_function([point.xc, point.yc,
                                     self.center.xc, self.center.yc,
                                     self.zeta], *self.parameters)
    return point.ix, point.iy, corrval

###################################
# Scipy Fitting Functions         #
###################################


def corotated_paraboloid_scipy(params, ucurv, vcurv, zoff):
    """
    Surface model to be used in fitting with scipy
    Same as the rotated paraboloid above, but theta is the panel center angle
    Not valid for polygon panels
    Assumes that the center of the paraboloid is the center of the panel
    Args:
        coords: [x,y] coordinate pair for point
        ucurv: curvature in projected u direction
        vcurv: curvature in projected v direction
        zoff:  Z offset of the paraboloid

    Returns:
    Paraboloid value at X and Y
    """
    xc, yc = params[0:2]
    x0, y0 = params[2:4]
    zeta = params[4]
    u = (xc - x0) * np.cos(zeta) - (yc - y0) * np.sin(zeta)
    v = (xc - x0) * np.sin(zeta) + (yc - y0) * np.cos(zeta)
    return ucurv * u**2 + vcurv * v**2 + zoff


def xyaxes_paraboloid_scipy(params, ucurv, vcurv, zoff):
    """
    Surface model to be used in fitting with scipy
    Assumes that the center of the paraboloid is the center of the panel
    In this model the panel can only bend in the x and y directions
    Args:
        coords: [x,y] coordinate pair for point
        ucurv: curvature in x direction
        vcurv: curvature in y direction
        zoff:  Z offset of the paraboloid

    Returns:
        Paraboloid value at X and Y
    """
    xc, yc = params[0:2]
    x0, y0 = params[2:4]
    u = xc - x0
    v = yc - y0
    return ucurv * u**2 + vcurv * v**2 + zoff


def rotated_paraboloid_scipy(params, ucurv, vcurv, zoff, theta):
    """
    Surface model to be used in fitting with scipy
    Assumes that the center of the paraboloid is the center of the panel
    This model is degenerate in the combinations of theta, ucurv and vcurv
    Args:
        coords: [x,y] coordinate pair for point
        ucurv: curvature in projected u direction
        vcurv: curvature in projected v direction
        zoff:  Z offset of the paraboloid
        theta: Angle between x,y and u,v coordinate systems

    Returns:
        Paraboloid value at X and Y
    """
    xc, yc = params[0:2]
    x0, y0 = params[2:4]
    u = (xc - x0) * np.cos(theta) - (yc - y0) * np.sin(theta)
    v = (xc - x0) * np.sin(theta) + (yc - y0) * np.cos(theta)
    return ucurv * u**2 + vcurv * v**2 + zoff


PANEL_MODEL_DICT = {
    "mean": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False,
        'fitting_function': None
    },
    "rigid": {
        'npar': 3,
        'solve': solve_rigid,
        'correct': correct_rigid,
        'experimental': False,
        'ring_only': False,
        'fitting_function': None
    },
    "flexible": {
        'npar': 4,
        'solve': solve_flexible,
        'correct': correct_flexible,
        'experimental': False,
        'ring_only': True,
        'fitting_function': None
    },
    "corotated_scipy": {
        'npar': 3,
        'solve': solve_scipy,
        'correct': correct_scipy,
        'experimental': False,
        'ring_only': False,
        'fitting_function': corotated_paraboloid_scipy
    },
    "corotated_lst_sq": {
        'npar': 3,
        'solve': solve_corotated_lst_sq,
        'correct': correct_corotated_lst_sq,
        'experimental': False,
        'ring_only': False,
        'fitting_function': None
    },
    "corotated_robust": {
        'npar': 3,
        'solve': solve_corotated_robust,
        'correct': correct_corotated_lst_sq,
        'experimental': False,
        'ring_only': False,
        'fitting_function': corotated_paraboloid_scipy
    },
    "xy_paraboloid": {
        'npar': 3,
        'solve': solve_scipy,
        'correct': correct_scipy,
        'experimental': False,
        'ring_only': False,
        'fitting_function': xyaxes_paraboloid_scipy
    },
    "rotated_paraboloid": {
        'npar': 4,
        'solve': solve_scipy,
        'correct': correct_scipy,
        'experimental': False,
        'ring_only': False,
        'fitting_function': rotated_paraboloid_scipy
    },
    "full_paraboloid_lst_sq": {
        'npar': 9,
        'solve': solve_full_paraboloid,
        'correct': correct_full_paraboloid,
        'experimental': True,
        'ring_only': False,
        'fitting_function': None
    }
}


class PanelModel:

    def __init__(self, model_dict, zeta, ref_points, center):
        self.zeta = zeta
        self.ref_points = ref_points
        self.center = center
        self.npar = model_dict['npar']
        self._solve = model_dict['solve']
        self._correct_point = model_dict['correct']
        self.fitting_function = model_dict['fitting_function']
        self.parameters = None
        self.fitted = False

    def _flexible_coeffs(self, point):
        x1, x2, y2 = self.ref_points
        f_lin = x1 + point.yc*(x2-x1)/y2
        coeffs = np.ndarray(4)
        coeffs[0] = (y2-point.yc) * (1.-point.xc/f_lin) / (2.0*y2)
        coeffs[1] = point.yc  * (1.-point.xc/f_lin) / (2.0*y2)
        coeffs[2] = (y2-point.yc) * (1.+point.xc/f_lin) / (2.0*y2)
        coeffs[3] = point.yc  * (1.+point.xc/f_lin) / (2.0*y2)
        return coeffs

    def solve(self, samples):
        self.parameters = self._solve(self, samples)
        self.fitted = True

    def correct(self, samples, margins):
        if not self.fitted:
            raise Exception("Cannot correct using a panel model that is not fitted")
        corrections = []
        for point in samples:
            corrections.append(self._correct_point(self, point))
        for point in margins:
            corrections.append(self._correct_point(self, point))
        return np.array(corrections)

    def correct_point(self, point):
        _, _, correction = self._correct_point(self, point)
        return correction


class PanelPoint:

    def __init__(self, xc, yc, ix=None, iy=None, value=None):
        self.xc = xc
        self.yc = yc
        self.ix = ix
        self.iy = iy
        self.value = value
