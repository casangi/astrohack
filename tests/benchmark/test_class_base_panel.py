import pytest

from astrohack._utils._panel_classes.base_panel import _gauss_elimination_numpy, BasePanel, \
     panel_models, imean, irigid, icorscp, icorlst, ixypara, icorrob, irotpara, ifulllst
from astrohack._utils._conversion import _convert_unit
import numpy as np

def gauss_elimination_numpy(size=3):
    """
    Tests the gaussian elimination routine by using an identity matrix
    """
    for pos in range(size):
        _gauss_elimination_numpy(np.identity(size), np.arange(size))[pos] == np.arange(size)[pos]

def test_gauss_elimination_numpy(benchmark):
    result = benchmark(gauss_elimination_numpy)

def add_point():
    """
    Test the add point common function
    """
    lepanel = BasePanel(panel_models[imean], np.zeros([4, 2]), np.zeros([4, 2]), 0.1, "TEST")
    for i in range(30):
        lepanel.add_sample([0, 0, 0, 0, 0])
        lepanel.add_margin([0, 0, 0, 0, 0])

def test_add_point(benchmark):
    result = benchmark(add_point)

