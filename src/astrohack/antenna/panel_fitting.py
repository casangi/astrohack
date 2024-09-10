import numpy as np
import scipy.optimize as opt

import toolviper.utils.logger as logger

from astrohack.utils import gauss_elimination, least_squares_jit


###################################
#  General purpose                #
###################################

def _fetch_sample_values(samples):
    points = np.ndarray((len(samples), 3))
    for ipnt, point in enumerate(samples):
        points[ipnt, :] = point.get_xcycval()
    return points


def _fetch_sample_coords(samples):
    coords = np.ndarray((len(samples), 4))
    for ipnt, point in enumerate(samples):
        coords[ipnt, :] = point.get_coords()
    return coords


def _fuse_idx_and_corrections(coords, corr1d):
    npnt = corr1d.shape[0]
    corrections = np.ndarray((npnt, 3))
    corrections[:, 0:2] = coords[:, 2:4]
    corrections[:, 2] = corr1d
    return corrections


def _build_system(shape):
    """
    Build a matrix and a vector to represent a system of linear equations
    Args:
        shape: The shape of the system [nrows, ncolumns]

    Returns:
        Nrows X ncolumns Matrix and nrows Vector
    """
    matrix = np.zeros(shape)
    vector = np.zeros([shape[0]])
    return matrix, vector

###################################
#  Mean                           #
###################################

def _solve_mean_opt(_self, samples):
    """
    Fit panel surface as a simple mean of its points deviation
    Args:
        _self: Parameter is here for consistent interface
        samples: List of points to be fitted

    Returns:
        the mean of the deviation in the points
    """
    mean = 0
    if len(samples) > 0:
        points = _fetch_sample_values(samples)
        mean = np.mean(points[:, 2])
    return [mean]


def _correct_mean_opt(self, points):
    """
    Provides the correction on a point using the mean value of the panel
    Args:
        self: The PanelModel object
        points: Points to be corrected

    Returns:
        The point indexes and the correction to that point.
    """
    return _fuse_idx_and_corrections(_fetch_sample_coords(points), np.full(len(points), self.parameters[0]))

###################################
# Rigid                           #
###################################

def _solve_rigid_opt(self, samples):
    """
    Fit panel surface using AIPS rigid model, an inclined plane.
    Args:
        self: The PanelModel object
        samples: List of points to be fitted

    Returns:
        The parameters for the fitted inclined plane
    """
    matrix, vector = _build_system([self.npar, self.npar])
    points = _fetch_sample_values(samples)
    matrix[0, 0] = np.sum(points[:, 0] ** 2)
    matrix[0, 1] = np.sum(points[:, 0] * points[:, 1])
    matrix[0, 2] = np.sum(points[:, 0])
    matrix[1, 0] = matrix[0, 1]
    matrix[1, 1] = np.sum(points[:, 1] * points[:, 1])
    matrix[1, 2] = np.sum(points[:, 1])
    matrix[2, 0] = matrix[0, 2]
    matrix[2, 1] = matrix[1, 2]
    matrix[2, 2] = len(samples)

    vector[0] = np.sum(points[:, 2] * points[:, 0])
    vector[1] = np.sum(points[:, 2] * points[:, 1])
    vector[2] = np.sum(points[:, 2])

    return gauss_elimination(matrix, vector)


def _correct_rigid_opt(self, points):
    """
    Provides the correction on a point using the fitted inclined plane
    Args:
        self: The PanelModel object
        points: Points to be corrected

    Returns:
        The point indexes and the correction to that point.
    """
    coords = _fetch_sample_coords(points)
    corr1d = coords[:, 0] * self.parameters[0] + coords[:, 1] * self.parameters[1] + self.parameters[2]
    return _fuse_idx_and_corrections(coords, corr1d)

###################################
# Flexible                        #
###################################

def _solve_flexible_opt(self, samples):
    """
    Fit panel surface using AIPS flexible model, WHAT IS THIS MODEL???
    Args:
        self: The PanelModel object
        samples: List of points to be fitted

    Returns:
        The parameters for the fitted model
    """
    matrix, vector = _build_system([self.npar, self.npar])
    coeffs_val = self._flexible_coeffs_arrays(samples)

    matrix[0, 0] = np.sum(coeffs_val[:, 0]*coeffs_val[:, 0])
    matrix[0, 1] = np.sum(coeffs_val[:, 0]*coeffs_val[:, 1])
    matrix[0, 2] = np.sum(coeffs_val[:, 0]*coeffs_val[:, 2])
    matrix[0, 3] = np.sum(coeffs_val[:, 0]*coeffs_val[:, 3])
    matrix[1, 1] = np.sum(coeffs_val[:, 1]*coeffs_val[:, 1])
    matrix[1, 2] = np.sum(coeffs_val[:, 1]*coeffs_val[:, 2])
    matrix[1, 3] = np.sum(coeffs_val[:, 1]*coeffs_val[:, 3])
    matrix[2, 2] = np.sum(coeffs_val[:, 2]*coeffs_val[:, 2])
    matrix[2, 3] = np.sum(coeffs_val[:, 2]*coeffs_val[:, 3])
    matrix[3, 3] = np.sum(coeffs_val[:, 3]*coeffs_val[:, 3])
    vector[0]   = np.sum(coeffs_val[:, 4]*coeffs_val[:, 0])
    vector[1]   = np.sum(coeffs_val[:, 4]*coeffs_val[:, 1])
    vector[2]   = np.sum(coeffs_val[:, 4]*coeffs_val[:, 2])
    vector[3]   = np.sum(coeffs_val[:, 4]*coeffs_val[:, 3])

    matrix[1, 0] = matrix[0, 1]
    matrix[2, 0] = matrix[0, 2]
    matrix[2, 1] = matrix[1, 2]
    matrix[3, 0] = matrix[0, 3]
    matrix[3, 1] = matrix[1, 3]
    matrix[3, 2] = matrix[2, 3]
    return gauss_elimination(matrix, vector)


def _correct_flexible_opt(self, points):
    """
    Provides the correction on a point using the fitted model
    Args:
        self: The PanelModel object
        points: Point to be corrected

    Returns:
        The point indexes and the correction to that point.
    """
    coeffs_val = self._flexible_coeffs_arrays(points)
    coeffs = coeffs_val[:, 0:4]
    coords = _fetch_sample_coords(points)
    corr1d = np.sum(coeffs[:, :] * self.parameters[:], axis=1)
    return _fuse_idx_and_corrections(coords, corr1d)


###################################
# Full 9 parameters paraboloid    #
###################################
def _solve_full_paraboloid_opt(self, samples):
    """
    Builds the designer matrix for least squares fitting, and calls the least_squares fitter for a fully fledged
    9 parameter paraboloid
    Args:
        self: The PanelModel object
        samples: List of points to be fitted

    Returns:
        The parameters for the fitted model
    """
    # ax2y2 + bx2y + cxy2 + dx2 + ey2 + gxy + hx + iy + j
    matrix, vector = _build_system((len(samples), self.npar))
    points = np.ndarray((len(samples), 3))
    for ipnt, point in enumerate(samples):
        points[ipnt, :] = point.get_xcycval()
    matrix[:, 0] = points[:, 0]**2 * points[:, 1]**2
    matrix[:, 1] = points[:, 0]**2 * points[:, 1]
    matrix[:, 2] = points[:, 1]**2 * points[:, 0]
    matrix[:, 3] = points[:, 0] ** 2
    matrix[:, 4] = points[:, 1] ** 2
    matrix[:, 5] = points[:, 0] * points[:, 1]
    matrix[:, 6] = points[:, 0]
    matrix[:, 7] = points[:, 1]
    matrix[:, 8] = 1.0
    vector[:] = points[:, 2]

    params, _, _, _ = least_squares_jit(matrix, vector)
    return params


def _correct_full_paraboloid_opt(self, points):
    """
    Provides the correction on a point using the fitted model
    Args:
        self: The PanelModel object
        points: Points to be corrected

    Returns:
        The point indexes and the correction to that point.
    """
    # ax2y2 + bx2y + cxy2 + dx2 + ey2 + gxy + hx + iy + j
    coords = _fetch_sample_coords(points)

    xx = coords[:, 0]
    yy = coords[:, 1]
    xsq = xx**2
    ysq = yy**2

    corr1d = (self.parameters[0]*xsq*ysq + self.parameters[1]*xsq*yy + self.parameters[2]*ysq*xx +
              self.parameters[3]*xsq + self.parameters[4]*ysq + self.parameters[5]*xx*yy + self.parameters[6]*xx +
              self.parameters[7]*yy + self.parameters[8])
    return _fuse_idx_and_corrections(coords, corr1d)

#######################################
# Co-rotated paraboloid least squares #
#######################################
def _solve_corotated_lst_sq_opt(self, samples):
    """
    Builds the designer matrix for least squares fitting, and calls the least_squares fitter for a corotated
    paraboloid centered at the center of the panel
    Args:
        self: The PanelModel object
        samples: List of points to be fitted

    Returns:
        The parameters for the fitted model
    """
    # a*u**2 + b*v**2 + c
    matrix, vector = _build_system((len(samples), self.npar))
    points = _fetch_sample_values(samples)
    x0 = self.center.xc
    y0 = self.center.yc

    matrix[:, 0] = ((points[:, 0] - x0) * np.cos(self.zeta) - (points[:, 1] - y0) * np.sin(self.zeta))**2  # U
    matrix[:, 1] = ((points[:, 0] - x0) * np.sin(self.zeta) + (points[:, 1] - y0) * np.cos(self.zeta))**2  # V
    matrix[:, 2] = 1.0
    vector[:] = points[:, 2]

    params, _, _, _ = least_squares_jit(matrix, vector)
    return params

def _correct_corotated_lst_sq_opt(self, points):
    """
    Provides the correction on a point using the fitted model
    Args:
        self: The PanelModel object
        points: Points to be corrected

    Returns:
        The point indexes and the correction to that point.
    """
    coords = _fetch_sample_coords(points)
    coszeta = np.cos(self.zeta)
    sinzeta = np.sin(self.zeta)
    x0 = self.center.xc
    y0 = self.center.yc
    corr1d = (((coords[:, 0] - x0) * coszeta - (coords[:, 1] - y0) * sinzeta) ** 2 * self.parameters[0] +
              ((coords[:, 0] - x0) * sinzeta + (coords[:, 1] - y0) * coszeta) ** 2 * self.parameters[1] +
              self.parameters[2])
    corrections = _fuse_idx_and_corrections(coords, corr1d)
    return corrections

###################################
# Co-rotated robust               #
###################################

def _solve_corotated_robust_opt(self, samples):
    """
    Try fitting the Surface of a panel using the corotated least_squares method, if that fails fallback to scipy
    fitting
    Args:
        self: The PanelModel object
        samples: List of points to be fitted

    Returns:
        The parameters for the fitted model
    """
    try:
        return _solve_corotated_lst_sq_opt(self, samples)
    except np.linalg.LinAlgError:
        return _solve_scipy_opt(self, samples)

###################################
# Scipy base                      #
###################################


def _solve_scipy_opt(self, samples, verbose=False, x0=None):
    """
    Fit the panel model using scipy optimiza curve_fit. The model is provided by a fitting function in the PanelModel
    object.
    Args:
        self: The PanelModel object
        samples: List of points to be fitted
        verbose: Print scipy fitting messages
        x0: user choice of initial values

    Returns:
        The parameters for the fitted model
    """

    coords = np.ndarray([5, len(samples)])
    points = _fetch_sample_values(samples)

    devia = points[:, 2]
    coords[0, :] = points[:, 0]
    coords[1, :] = points[:, 1]
    coords[2, :] = self.center.xc
    coords[3, :] = self.center.yc
    coords[4, :] = self.zeta

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
            result = opt.curve_fit(self._fitting_function, coords, devia,
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


def _correct_scipy_opt(self, points):
    corrections = np.array([[point.ix, point.iy,
                             self._fitting_function([point.xc, point.yc, self.center.xc, self.center.yc, self.zeta],
                                                    *self.parameters)]
                            for point in points])
    return corrections

###################################
# Scipy Fitting Functions         #
###################################


def _corotated_paraboloid_scipy(params, ucurv, vcurv, zoff):
    """
    Fitting function for a corrotated paraboloid to be used with solve_scipy
    Args:
        params: [xc, yc, x0, y0, zeta] Coordinates and non-fitted model parameters
        ucurv: curvature in projected u direction
        vcurv: curvature in projected v direction
        zoff:  Z offset of the paraboloid

    Returns:
    Paraboloid value at Xc and Yc
    """
    xc, yc = params[0:2]
    x0, y0 = params[2:4]
    zeta = params[4]
    u = (xc - x0) * np.cos(zeta) - (yc - y0) * np.sin(zeta)
    v = (xc - x0) * np.sin(zeta) + (yc - y0) * np.cos(zeta)
    return ucurv * u**2 + vcurv * v**2 + zoff


def _xyaxes_paraboloid_scipy(params, ucurv, vcurv, zoff):
    """
    Fitting function for a simple paraboloid to be used with solve_scipy whose bending axes are parallel to the
    X and Y directions.
    Args:
        params: [xc, yc, x0, y0, zeta] Coordinates and non-fitted model parameters
        ucurv: curvature in projected u direction
        vcurv: curvature in projected v direction
        zoff:  Z offset of the paraboloid

    Returns:
    Paraboloid value at Xc and Yc
    """
    xc, yc = params[0:2]
    x0, y0 = params[2:4]
    u = xc - x0
    v = yc - y0
    return ucurv * u**2 + vcurv * v**2 + zoff


def _rotated_paraboloid_scipy(params, ucurv, vcurv, zoff, theta):
    """
    Fitting function for a simple paraboloid to be used with solve_scipy whose bending axes can be arbitrarily rotated
    from the X and Y axes
    Args:
        params: [xc, yc, x0, y0, zeta] Coordinates and non-fitted model parameters
        ucurv: curvature in projected u direction
        vcurv: curvature in projected v direction
        zoff:  Z offset of the paraboloid
        theta: Paraboloids rotation with respect to the X axis

    Returns:
    Paraboloid value at Xc and Yc
    """
    xc, yc = params[0:2]
    x0, y0 = params[2:4]
    u = (xc - x0) * np.cos(theta) - (yc - y0) * np.sin(theta)
    v = (xc - x0) * np.sin(theta) + (yc - y0) * np.cos(theta)
    return ucurv * u**2 + vcurv * v**2 + zoff


PANEL_MODEL_DICT = {
    "mean": {
        'npar': 1,
        'solve': _solve_mean_opt,
        'correct': _correct_mean_opt,
        'experimental': False,
        'ring_only': False,
        'fitting_function': None
    },
    "rigid": {
        'npar': 3,
        'solve': _solve_rigid_opt,
        'correct': _correct_rigid_opt,
        'experimental': False,
        'ring_only': False,
        'fitting_function': None
    },
    "flexible": {
        'npar': 4,
        'solve': _solve_flexible_opt,
        'correct': _correct_flexible_opt,
        'experimental': False,
        'ring_only': True,
        'fitting_function': None
    },
    "corotated_scipy": {
        'npar': 3,
        'solve': _solve_scipy_opt,
        'correct': _correct_corotated_lst_sq_opt,
        'experimental': False,
        'ring_only': False,
        'fitting_function': _corotated_paraboloid_scipy
    },
    "corotated_lst_sq": {
        'npar': 3,
        'solve': _solve_corotated_lst_sq_opt,
        'correct': _correct_corotated_lst_sq_opt,
        'experimental': False,
        'ring_only': False,
        'fitting_function': None
    },
    "corotated_robust": {
        'npar': 3,
        'solve': _solve_corotated_robust_opt,
        'correct': _correct_corotated_lst_sq_opt,
        'experimental': False,
        'ring_only': False,
        'fitting_function': _corotated_paraboloid_scipy
    },
    "xy_paraboloid": {
        'npar': 3,
        'solve': _solve_scipy_opt,
        'correct': _correct_scipy_opt,
        'experimental': False,
        'ring_only': False,
        'fitting_function': _xyaxes_paraboloid_scipy
    },
    "rotated_paraboloid": {
        'npar': 4,
        'solve': _solve_scipy_opt,
        'correct': _correct_scipy_opt,
        'experimental': False,
        'ring_only': False,
        'fitting_function': _rotated_paraboloid_scipy
    },
    "full_paraboloid_lst_sq": {
        'npar': 9,
        'solve': _solve_full_paraboloid_opt,
        'correct': _correct_full_paraboloid_opt,
        'experimental': True,
        'ring_only': False,
        'fitting_function': None
    },
}


class PanelModel:

    def __init__(self, model_dict, zeta, ref_points, center):
        """
        Initialize a PanelModel object
        Args:
            model_dict: The dictionary containing the parameters of the model
            zeta: The panel angle from the Y axis (only used for some models)
            ref_points: Reference points for use in the flexible model fitting
            center: Panel center (only used for some models)
        """
        self.zeta = zeta
        self.ref_points = ref_points
        self.center = center
        self.npar = model_dict['npar']
        self._solve = model_dict['solve']
        self._correct_sub = model_dict['correct']
        self._fitting_function = model_dict['fitting_function']
        self.parameters = None
        self.fitted = False

    def _flexible_coeffs_arrays(self, samples):
        points = _fetch_sample_values(samples)
        coeffs_val = np.ndarray((len(samples), 5))

        x1, x2, y2 = self.ref_points
        f_lin = x1 + points[:, 1]*(x2-x1)/y2
        coeffs_val[:, 0] = (y2-points[:, 1]) * (1.-points[:, 0]/f_lin) / (2.0*y2)
        coeffs_val[:, 1] = points[:, 1]  * (1.-points[:, 0]/f_lin) / (2.0*y2)
        coeffs_val[:, 2] = (y2-points[:, 1]) * (1.+points[:, 0]/f_lin) / (2.0*y2)
        coeffs_val[:, 3] = points[:, 1]  * (1.+points[:, 0]/f_lin) / (2.0*y2)
        coeffs_val[:, 4] = points[:, 2]

        return coeffs_val

    def solve(self, samples):
        """
        Fits the model to the given samples
        Args:
            samples: The list of points to be fitted.

        Returns:
            Nothing
        """
        self.parameters = self._solve(self, samples)
        self.fitted = True

    def correct(self, samples, margins):
        """
        Provides the corrections for all the points in the margins and samples
        Args:
            samples: The list of points to be fitted.
            margins: The list of points to be ignored in fitting but used in corrections.

        Returns:
            Array of corrections and the indices linking them to the aperture.
        """
        if not self.fitted:
            raise Exception("Cannot correct using a panel model that is not fitted")

        nsamp = len(samples)
        nmarg = len(margins)

        if nsamp == 0 and nmarg == 0:
            raise Exception("Nothing to correct")
        elif nmarg == 0:
            corrections = np.array(self._correct_sub(self, samples))
        elif nsamp == 0:
            corrections = np.array(self._correct_sub(self, margins))
        else:
            samp_corr = np.array(self._correct_sub(self, samples))
            marg_corr = np.array(self._correct_sub(self, margins))
            corrections = np.concatenate((samp_corr, marg_corr), axis=0)

        return corrections

    def correct_point(self, point):
        """
        Provide corrections for a single PanelPoint
        Args:
            point: the point in question

        Returns:
            The expected correction for that single point.
        """
        correction = self._correct_sub(self, [point])
        return correction[0, 2]


class PanelPoint:

    def __init__(self, xc, yc, ix=None, iy=None, value=None):
        """
        Initialize a point with its important properties
        Args:
            xc: X coordinate
            yc: Y coordinate
            ix: x-axis index at the aperture if relevant
            iy: y-axis index at the aperture if relevant
            value: point value if relevant
        """
        self.xc = xc
        self.yc = yc
        self.ix = ix
        self.iy = iy
        self.value = value

    def __eq__(self, other):
        equal = self.xc == other.xc
        equal = equal and self.yc == other.yc
        equal = equal and self.ix == other.ix
        equal = equal and self.iy == other.iy
        equal = equal and self.value == other.value
        return equal

    def get_xcycval(self):
        return self.xc, self.yc, self.value

    def get_coords(self):
        return self.xc, self.yc, self.ix, self.iy


