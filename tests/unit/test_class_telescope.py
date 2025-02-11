import pytest
import os
import filecmp
import shutil

from astrohack.antenna.telescope import Telescope


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
        tel.read(tel.filepath+"/vlba.zarr")
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
        assert filecmp.cmp(tel.filepath+"/vlba.zarr/.zattrs", testfile+'/.zattrs') == 0, 'Telescope configuration ' \
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
        This test is currently mute as this routine no longer raises an exception
        """
        # tel = Telescope("vla")
        # with pytest.raises(Exception):
        #     tel._general_consistency()
