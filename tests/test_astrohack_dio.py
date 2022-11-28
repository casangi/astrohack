import os
import gdown
import tarfile
import pytest

import numpy as np

from astrohack import __version__
from astrohack.dio import load_hack_file
from astrohack._utils._io import _load_pnt_dict


def test_load_hack_file():
    """ Simple test of hackfile loading functionality.

            This function downloads a reference hackfile and checks that the dimensions
            of the loaded file match the known reference values.
    """

    url = 'https://drive.google.com/u/0/uc?id=1PqLmoDXzv8vl0rXWTMTyYRyJuj6E1fCD'
    output = 'hackfile.tar'

    gdown.download(url, output)

    tar = tarfile.open( "/".join((os.getcwd(), output)) )
    tar.extractall()
    tar.close()

    filename = "/".join((os.getcwd(), "hackfile"))

    hackfile = load_hack_file(filename)
    
    test_pnt_dict_dims = {'time': 60072, 'az_el': 2, 'ra_dec': 2}
    test_data_dims = {'time': 2401, 'lm': 2, 'chan': 64, 'pol': 4}

    assert hackfile['pnt_dict'][0].dims == test_pnt_dict_dims, "load_hack_file::Pointing dict dimensions do not match."
    assert hackfile[0][2][19].dims == test_data_dims, "load_hack_file::Data dimensions do not match."

def test_load_pnt_dict():
    """ Simple test of pointing dictionary loading functionality.

            This function downloads a reference pointing dictionary and checks that the dimensions
            of the loaded file match the known reference values.
    """

    url = 'https://drive.google.com/u/0/uc?id=1d7ARSgO2DwmXSUf7T1IbC8t5yYa3dbGr'
    output = 'test.pnt.dict.tar'

    gdown.download(url, output)
 
    tar = tarfile.open( "/".join((os.getcwd(), output)) )
    tar.extractall()
    tar.close()

    pnt_dict = _load_pnt_dict("/".join((os.getcwd(), "test.pnt.dict")))
    
    test_pnt_dict_dims = {'time': 60072, 'az_el': 2, 'ra_dec': 2}

    assert pnt_dict[0].dims == test_pnt_dict_dims, "load_pnt_dict::Pointing dict dimensions do not match."

def test_version():
    assert __version__ == '0.1.0'
