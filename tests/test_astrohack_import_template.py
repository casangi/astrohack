import pytest

@classmethod
def setup_class(cls):
    """ setup any state specific to the execution of the given test class
    such as fetching test data """
    pass

@classmethod
def teardown_class(cls):
    """ teardown any state that was previously setup with a call to setup_class
    such as deleting test data """
    pass

def setup():
    """ setup any state specific to all methods of the given class """
    pass

def teardown():
    """ teardown any state that was previously setup for all methods of the given class """
    pass

def test_import_astrohack_client():
  "Import astrohack_client"
    try:
        from astrohack.astrohack_client import astrohack_local_client
    except ImportError:
        assert False

def test_import_extract_holog():
    try:
        from astrohack.extract_holog import extract_holog
    except ImportError:
        assert False

def test_import_holog():
    try:
        from astrohack.holog import holog
    except ImportError:
        assert False

def test_import_panel():
    try:
        from astrohack.panel import panel
    except ImportError:
        assert False

def test_import_dio_open_holog():
    try:
        from astrohack.dio import open_holog
    except ImportError:
        assert False

def test_import_dio_open_image():
    try:
        from astrohack.dio import open_image
    except ImportError:
        assert False

def test_import_dio_open_panel():
    try:
        from astrohack.dio import open_panel
    except ImportError:
        assert False
 
def test_import_dio_open_pointing():
    try:
        from astrohack.dio import open_pointing
    except ImportError:
        assert False

def test_import_dio_fix_pointing_table():
    try:
        from astrohack.dio import fix_pointing_table
    except ImportError:
        assert False
        
def test_import_dio_export_screws():
    try:
        from astrohack.dio import export_screws
    except ImportError:
        assert False
        
def test_import_dio_plot_antenna():
    try:
        from astrohack.dio import plot_antenna
    except ImportError:
        assert False
        
def test_import_dio_export_to_fits():
    try:
        from astrohack.dio import export_to_fits
    except ImportError:
        assert False
        
        
