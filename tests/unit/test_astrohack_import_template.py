import pytest

class TestAstrohack():
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

    def setup_method(self):
        """ setup any state specific to all methods of the given class """
        pass

    def teardown_method(self):
        """ teardown any state that was previously setup for all methods of the given class """
        pass

    def test_import_astrohack_client(self):
        """ Import astrohack_client """
        try:
            from astrohack.astrohack_client import astrohack_local_client
        except ImportError:
            assert False

    def test_import_extract_holog(self):
        try:
            from astrohack.extract_holog import extract_holog
        except ImportError:
            assert False

    def test_import_holog(self):
        try:
            from astrohack.holog import holog
        except ImportError:
            assert False

    def test_import_panel(self):
        try:
            from astrohack.panel import panel
        except ImportError:
            assert False

    def test_import_dio_open_holog(self):
        try:
            from astrohack.dio import open_holog
        except ImportError:
            assert False

    def test_import_dio_open_image(self):
        try:
            from astrohack.dio import open_image
        except ImportError:
            assert False

    def test_import_dio_open_panel(self):
        try:
            from astrohack.dio import open_panel
        except ImportError:
            assert False

    def test_import_dio_open_pointing(self):
        try:
            from astrohack.dio import open_pointing
        except ImportError:
            assert False

    def test_import_dio_fix_pointing_table(self):
        try:
            from astrohack.dio import fix_pointing_table
        except ImportError:
            assert False

    def test_import_dio_print_json(self):
        try:
            from astrohack.dio import print_json
        except ImportError:
            assert False

    def test_import_dio_inspect_holog_obs_dict(self):
        try:
            from astrohack.dio import inspect_holog_obs_dict
        except ImportError:
            assert False
