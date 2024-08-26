import numpy as np

###################################
###  MEAN                       ###
###################################

def solve_mean(samples):
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


PANEL_MODEL_DICT = {
    "mean": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    },
    "rigid": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    },
    "flexible": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    },
    "corotated_scipy": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    },
    "corotated_lst_sq": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    },
    "corotated_robust": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    },
    "xy_paraboloid": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    },
    "rotated_paraboloid": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    },
    "full_paraboloid_lst_sq": {
        'npar': 1,
        'solve': solve_mean,
        'correct': correct_mean,
        'experimental': False,
        'ring_only': False
    }
}
