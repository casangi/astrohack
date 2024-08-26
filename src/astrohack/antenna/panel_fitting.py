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
###  MEAN                       ###
###################################

def solve_mean(samples, _model_dict):
    """
    Fit panel surface as a simple mean of its points Z deviation
    """
    if len(samples) > 0:
        # Solve panel adjustments for rigid vertical shift only panels
        data = np.array(samples)[:, -1]
        par = [np.mean(data)]
    else:
        par = [0]
    return par

def correct_mean(_xc, _yc, par):
    return par[0]

###################################
###  RIGID                      ###
###################################

def solve_rigid(samples):
    """
    Fit panel surface using AIPS gaussian elimination model for rigid panels
    """
    npar = 3
    matrix, vector = build_system(npar)
    for point in samples:
        matrix[0, 0] += point[0] * point[0]
        matrix[0, 1] += point[0] * point[1]
        matrix[0, 2] += point[0]
        matrix[1, 0] = matrix[0, 1]
        matrix[1, 1] += point[1] * point[1]
        matrix[1, 2] += point[1]
        matrix[2, 0] = matrix[0, 2]
        matrix[2, 1] = matrix[1, 2]
        matrix[2, 2] += 1.0
        vector[0] += point[-1] * point[0]
        vector[1] += point[-1] * point[1]
        vector[2] += point[-1]

    return gauss_elimination(matrix, vector)

def correct_rigid(xc, yc, par):
    """
    Computes fitted value for point [xcoor, ycoor] using AIPS gaussian elimination model for rigid panels
    Args:
        xc: X coordinate of point
        yc: Y coordinate of point

    Returns:
    Fitted value at xcoor,ycoor
    """
    return xc * par[0] + yc * par[1] + par[2]



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
    # "flexible": {
    #     'npar': 1,
    #     'solve': solve_mean,
    #     'correct': correct_mean,
    #     'experimental': False,
    #     'ring_only': False
    # },
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
