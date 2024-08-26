import numpy as np

from astrohack.utils import gauss_elimination


###################################
###  General purpose            ###
###################################

def build_system(size):
    matrix = np.zeros([size, size])
    vector = np.zeros([size])
    return matrix, vector



###################################
###  Mean                       ###
###################################

def solve_mean(samples, _ref_points):
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

def correct_mean(_xc, _yc, model_fit, _ref_points):
    return model_fit[0]

###################################
###  Rigid                      ###
###################################

def solve_rigid(samples, _ref_points):
    """
    Fit panel surface using AIPS gaussian elimination model for rigid panels
    """
    npar = 3
    matrix, vector = build_system(npar)
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

def correct_rigid(xc, yc, model_fit, _ref_points):
    """
    Computes fitted value for point [xcoor, ycoor] using AIPS gaussian elimination model for rigid panels
    Args:
        xc: X coordinate of point
        yc: Y coordinate of point

    Returns:
    Fitted value at xcoor,ycoor
    """
    return xc * model_fit[0] + yc * model_fit[1] + model_fit[2]

###################################
###  Flexible                   ###
###################################

def _flexible_coeffs(xc, yc, ref_points):
    x1, x2, y2 = ref_points
    f_lin = x1 + yc*(x2-x1)/y2
    coeffs = np.ndarray(4)
    coeffs[0] = (y2-yc) * (1.-xc/f_lin) / (2.0*y2)
    coeffs[1] =     yc  * (1.-xc/f_lin) / (2.0*y2)
    coeffs[2] = (y2-yc) * (1.+xc/f_lin) / (2.0*y2)
    coeffs[3] =     yc  * (1.+xc/f_lin) / (2.0*y2)
    return coeffs

def solve_flexible(samples, ref_points):
    # this can only work for ringed panels...
    system, vector = build_system(4)
    for point in samples:
        auno, aduo, atre, aqua = _flexible_coeffs(point.xc, point.yc, ref_points)
        system[0,0] += auno*auno
        system[0,1] += auno*aduo
        system[0,2] += auno*atre
        system[0,3] += auno*aqua
        system[1,1] += aduo*aduo
        system[1,2] += aduo*atre
        system[1,3] += aduo*aqua
        system[2,2] += atre*atre
        system[2,3] += atre*aqua
        system[3,3] += aqua*aqua
        vector[0]   += point.value*auno
        vector[1]   += point.value*aduo
        vector[2]   += point.value*atre
        vector[3]   += point.value*aqua

    system[1,0] = system[0,1]
    system[2,0] = system[0,2]
    system[2,1] = system[1,2]
    system[3,0] = system[0,3]
    system[3,1] = system[1,3]
    system[3,2] = system[2,3]
    return gauss_elimination(system, vector)

def correct_flexible(xc, yc, model_fit, ref_points):
    coeffs = _flexible_coeffs(xc, yc, ref_points)
    return np.sum(coeffs * np.array(model_fit))




PANEL_MODEL_DICT = {
    "mean": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    },
    "rigid": {
        'npar': 3,
        'solve':solve_rigid,
        'correct': correct_rigid,
        'experimental': False,
        'ring_only': False
    },
    "flexible": {
        'npar': 4,
        'solve': solve_flexible,
        'correct': correct_flexible,
        'experimental': False,
        'ring_only': False
    },
    # "corotated_scipy": {
    #     'npar': 1,
    #     'solve': solve_mean,
    #     'correct': correct_mean,
    #     'experimental': False,
    #     'ring_only': False
    # },
    # "corotated_lst_sq": {
    #     'npar': 1,
    #     'solve': solve_mean,
    #     'correct': correct_mean,
    #     'experimental': False,
    #     'ring_only': False
    # },
    # "corotated_robust": {
    #     'npar': 1,
    #     'solve': solve_mean,
    #     'correct': correct_mean,
    #     'experimental': False,
    #     'ring_only': False
    # },
    # "xy_paraboloid": {
    #     'npar': 1,
    #     'solve': solve_mean,
    #     'correct': correct_mean,
    #     'experimental': False,
    #     'ring_only': False
    # },
    # "rotated_paraboloid": {
    #     'npar': 1,
    #     'solve': solve_mean,
    #     'correct': correct_mean,
    #     'experimental': False,
    #     'ring_only': False
    # },
    # "full_paraboloid_lst_sq": {
    #     'npar': 1,
    #     'solve': solve_mean,
    #     'correct': correct_mean,
    #     'experimental': False,
    #     'ring_only': False
    # }
}


class PanelPoint:

    def __init__(self, xc, yc, ix, iy, value):
        self.xc = xc
        self.yc = yc
        self.ix = ix
        self.iy = iy
        self.value = value
