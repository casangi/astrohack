import pytest
from astrohack._classes.telescope import Telescope, _find_cfg_file, tel_data_path
import os
import filecmp
import shutil


class TestClassTelescope:
    def test_init(self):
        """
        Test the initialization of a Telescope object using the VLA as a test case
        """
        tel = Telescope('vla')
        assert tel.name == 'VLA', 'Telescope name loaded incorrectly'
        assert tel.diam == 25.0, 'Telescope diameter loaded incorrectly'

        with pytest.raises(Exception):
            tel = Telescope("xxx")

    def test_read(self):
        """
        Tests the reading of a hack file and the errors when trying to read a non-existent file
        """
        tel = Telescope('vla')
        tel.read(tel_data_path+'/vlba.zarr')
        assert tel.name == 'VLBA', 'Telescope name loaded incorrectly'
        assert tel.focus == 8.75, 'Telescope focus length loaded incorrectly'

        with pytest.raises(FileNotFoundError):
            tel.read("xxx")

    def test_write(self):
        """
        Test the writting of a hack file containing the telescope atributes
        """
        testfile = "test-tel.zarr"
        tel = Telescope("vla")
        tel.write(testfile)
        assert os.path.exists(testfile), 'Telescope configuration file not created at the proper location'
        assert filecmp.cmp(tel_data_path+'/vlba.zarr/.zattrs', testfile+'/.zattrs') == 0, 'Telescope configuration ' \
                                                                                          'file is not equal to the ' \
                                                                                          'reference'
        shutil.rmtree(testfile)

    def test_ringed_consistency(self):
        """
        Tests the consistency checks on ringed layout Telescope object
        """
        tel = Telescope("vla")
        tel.onaxisoptics = False
        with pytest.raises(Exception):
            tel._ringed_consistency()
        tel.nrings = 1000
        with pytest.raises(Exception):
            tel._ringed_consistency()

    def test_general_consistency(self):
        """
        Tests the consistency on a general layout Telescope Object
        """
        tel = Telescope("vla")
        with pytest.raises(Exception):
            tel._general_consistency()

    def test_find_cfg_file(self):
        """
        tests the routine to automatically find a hack cfg file for a Telescope object
        """
        filefullpath = _find_cfg_file('vla.zarr', tel_data_path)
        assert filefullpath == tel_data_path+'/vla.zarr', 'Returned path is not the expected path'
        with pytest.raises(FileNotFoundError):
            dummy = _find_cfg_file("xxx", "./xxx")
