import pytest
import os
import filecmp
import shutil

from astrohack import get_proper_telescope
from astrohack.antenna.telescope import Telescope, RingedCassegrain, NgvlaPrototype


class TestClassTelescope:
    def test_get_proper_telescope(self):
        """
        Test the initialization of a Telescope object using the VLA as a test case
        """
        tel = get_proper_telescope("vla")
        assert isinstance(tel, RingedCassegrain), "Telescope initialized to the proper class"
        assert tel.name == "VLA", "Telescope name loaded incorrectly"
        assert tel.diameter == 25.0, "Telescope diameter loaded incorrectly"

        with pytest.raises(Exception):
            vla_ns = get_proper_telescope("vla", 'teletubies')

        ngvla = get_proper_telescope("VLA", "na")
        assert ngvla is None, "ngvla telescopes are not initializable yet"

        alma_da = get_proper_telescope("ALMA", "DA13")
        assert alma_da.name == "ALMA DA", "ALMA DA is not properly initialized"

        alma_dv = get_proper_telescope("ALMA", "DV13")
        assert alma_dv.name == "ALMA DV", "ALMA DV is not properly initialized"

        alma_tp = get_proper_telescope("ALMA", "TP3")
        assert alma_tp.name == "ALMA TP", "ALMA TP is not properly initialized"

        with pytest.raises(Exception):
            alma_ns = get_proper_telescope("ALMA")

        with pytest.raises(Exception):
            alma_ns = get_proper_telescope("ALMA", 'teletubies')

        newtel = get_proper_telescope('teletubies')
        assert newtel is None, "Nonsense telescope name does not return None"

    def test_read(self):
        """
        Tests the reading of a hack file and the errors when trying to read a non-existent file
        """
        tel = get_proper_telescope("vla")
        tel.read(tel.file_path + "/vlba.zarr")
        assert tel.name == "VLBA", "Telescope name loaded incorrectly"
        assert tel.focus == 8.75, "Telescope focus length loaded incorrectly"

        with pytest.raises(FileNotFoundError):
            tel.read("teletubies")

    def test_write(self):
        """
        Test the writting of a hack file containing the telescope atributes
        """
        testfile = "teletubies-tel.zarr"
        tel = get_proper_telescope("vla")
        tel.write(testfile)
        assert os.path.exists(
            testfile
        ), "Telescope configuration file not created at the proper location"
        assert (
            filecmp.cmp(tel.file_path + "/vlba.zarr/.zattrs", testfile + "/.zattrs")
            == 0
        ), "Telescope configuration " "file is not equal to the " "reference"
        shutil.rmtree(testfile)

        tel.name = 'teletubies'
        tel.write_to_distro()
        assert os.path.exists(
            tel.file_path + '/teletubies.zarr'
        ), "Telescope configuration file not created at the proper location"
        shutil.rmtree(tel.file_path + '/teletubies.zarr')

    def test_ringed_consistency(self):
        """
        Tests the consistency checks on ringed layout Telescope object
        """
        tel = get_proper_telescope("vla")
        # tel.onaxisoptics = False
        # with pytest.raises(Exception):
        #     tel._ringed_consistency()
        # tel.nrings = 1000
        # with pytest.raises(Exception):
        #     tel._ringed_consistency()

    def test_build_panel_list(self):
        return

    def test_assign_panel(self):
        return

    def test_general_consistency(self):
        """
        Tests the consistency on a general layout Telescope Object
        This test is currently mute as this routine no longer raises an exception
        """
        # tel = Telescope("vla")
        # with pytest.raises(Exception):
        #     tel._general_consistency()
