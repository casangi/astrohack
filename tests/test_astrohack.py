import numpy as np

from astrohack import __version__
from astrohack.dio import load_hack_file


@pytest.fixture
def reference_hack_file():
    """ Build a reference hakfile to compare against.

    Returns:
        hack(dict): Reference hackfile
    """
    
    return None

def test_load_hack_file():
    return None

def test_version():
    assert __version__ == '0.0.1'
