import numpy as np


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
