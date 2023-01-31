import numpy as np


def _gauss_elimination_numpy(system, vector):
    """
    Gauss elimination solving of a system using numpy
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system

    Returns:
    The solved system
    """
    inverse = np.linalg.inv(system)
    return np.dot(inverse, vector)


def _least_squares_fit(system, vector):
    """
    Least squares fitting of a system of linear equations
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system

    Returns:
    The solved system, the variances of the system solution and the sum of the residuals
    """
    if len(system.shape) != 2:
        raise Exception('System must have 2 dimensions')
    if system.shape[0] != system.shape[1]:
        raise Exception('System must be a square matrix')
    rank = system.shape[0]
    fit = np.linalg.lstsq(system, vector)
    result = fit[0]
    residuals = fit[1]
    covar = np.matrix(np.dot(system.T, system)).I
    variances = np.dot(covar, np.identity(rank))
    return result, variances, residuals
