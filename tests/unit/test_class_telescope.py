import pytest
import os
import filecmp
import shutil
import numpy as np
from shapely.geometry.polygon import Polygon

from astrohack.utils.ray_tracing_general import simple_axis
from astrohack.antenna import RingPanel
from astrohack.antenna.telescope import RingedCassegrain, get_proper_telescope


class TestClassTelescope:
    def test_get_proper_telescope(self):
        """
        Test the initialization of a Telescope object using the VLA as a test case
        """
        tel = get_proper_telescope("vla")
        assert isinstance(
            tel, RingedCassegrain
        ), "Telescope initialized to the proper class"
        assert tel.name == "VLA", "Telescope name loaded incorrectly"
        assert tel.diameter == 25.0, "Telescope diameter loaded incorrectly"

        with pytest.raises(Exception):
            vla_ns = get_proper_telescope("vla", "teletubies")

        ngvla = get_proper_telescope("VLA", "na")
        assert (
            ngvla.name == "ngVLA proto 2025"
        ), "ngVLA prototype is not properly initialized"

        alma_da = get_proper_telescope("ALMA", "DA13")
        assert alma_da.name == "ALMA DA", "ALMA DA is not properly initialized"

        alma_dv = get_proper_telescope("ALMA", "DV13")
        assert alma_dv.name == "ALMA DV", "ALMA DV is not properly initialized"

        alma_tp = get_proper_telescope("ALMA", "TP3")
        assert alma_tp.name == "ALMA TP", "ALMA TP is not properly initialized"

        with pytest.raises(Exception):
            alma_ns = get_proper_telescope("ALMA")

        with pytest.raises(Exception):
            alma_ns = get_proper_telescope("ALMA", "teletubies")

        newtel = get_proper_telescope("teletubies")
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
        ), ("Telescope configuration " "file is not equal to the " "reference")
        shutil.rmtree(testfile)

        tel.name = "teletubies"
        tel.write_to_distro()
        assert os.path.exists(
            tel.file_path + "/teletubies.zarr"
        ), "Telescope configuration file not created at the proper location"
        shutil.rmtree(tel.file_path + "/teletubies.zarr")

    def test_ringed_consistency(self):
        """
        Tests the consistency checks on ringed layout Telescope object
        """
        tel = get_proper_telescope("vla")
        tel.consistency_check()
        tel.diameter = 10
        with pytest.raises(Exception):
            tel.consistency_check()

        tel = get_proper_telescope("vla")
        tel.n_rings_of_panels = 1000
        with pytest.raises(Exception):
            tel.consistency_check()

    def test_create_ringed_aperture_mask(self):
        tel = get_proper_telescope("vla")
        u_axis = simple_axis([-15, 15], 0.1, 0.0)
        v_axis = simple_axis([-15, 15], 0.1, 0.0)
        mid_point = u_axis.shape[0] // 2

        mask = tel.create_aperture_mask(u_axis, v_axis)
        assert mask[130, 130], "Mask should be True here"
        assert not mask[mid_point, mid_point], "Mask should be False inside blockage"
        assert not mask[130, 150], "Mask should be False in an arm shadow"

        mask = tel.create_aperture_mask(u_axis, v_axis, use_detailed_mask=False)
        assert mask[130, 130], "Mask should be True here"
        assert not mask[mid_point, mid_point], "Mask should be False inside blockage"
        assert mask[130, 150], "Mask should be True in an arm shadow"

    def test_create_ngvla_aperture_mask(self):
        tel = get_proper_telescope("ngvla")
        u_axis = simple_axis([-10, 10], 0.1, 0.0)
        v_axis = simple_axis([-10, 10], 0.1, 0.0)
        mid_point = u_axis.shape[0] // 2

        mask = tel.create_aperture_mask(u_axis, v_axis, use_detailed_mask=False)
        radius = np.sqrt(u_axis[np.newaxis, :] ** 2 + v_axis[:, np.newaxis] ** 2)
        ref_mask = np.where(radius <= tel.diameter / 2, True, False)
        assert mask[
            mid_point, mid_point
        ], "Mask should be true at the center of an unblocked aperture"
        assert np.all(ref_mask == mask), "Mask is not identical to reference"

        mask = tel.create_aperture_mask(u_axis, v_axis, use_detailed_mask=True)
        for panel_label, panel_data in tel.panel_dict.items():
            poly = Polygon(tel.panel_dict[panel_label]["polygon"])
            iu = np.argmin(np.abs(u_axis - poly.centroid.x))
            iv = np.argmin(np.abs(v_axis - poly.centroid.y))
            assert mask[
                iu, iv
            ], f"Panel {panel_label} centroid should be included in mask"

    def test_build_ringed_panel_list(self):
        tel = get_proper_telescope("vla")
        panel_list = tel.build_panel_list("flexible", 0.2)
        assert isinstance(
            panel_list[0], RingPanel
        ), "Wrong class for panels in panel list"
        assert (
            len(panel_list) == 172
        ), "Panel list for the VLA is produced with the wrong number of panels"
        assert (
            panel_list[0].label == "1-1"
        ), "Labelling for VLA panels is not working as expected"
        assert (
            panel_list[134].label == "6-3"
        ), "Labelling for VLA panels is not working as expected"
        assert (
            panel_list[-1].label == "6-40"
        ), "Labelling for VLA panels is not working as expected"

        tel = get_proper_telescope("alma", "dv12")
        panel_list = tel.build_panel_list("flexible", 0.2)
        assert isinstance(
            panel_list[0], RingPanel
        ), "Wrong class for panels in panel list"
        assert (
            len(panel_list) == 264
        ), "Panel list for the ALMA DV is produced with the wrong number of panels"
        assert (
            panel_list[0].label == "3-11"
        ), "Labelling for ALMA DV panels is not working as expected"
        assert (
            panel_list[153].label == "7-63"
        ), "Labelling for ALMA DV panels is not working as expected"
        assert (
            panel_list[-1].label == "4-81"
        ), "Labelling for ALMA DV panels is not working as expected"

        tel = get_proper_telescope("alma", "da51")
        panel_list = tel.build_panel_list("flexible", 0.2)
        assert isinstance(
            panel_list[0], RingPanel
        ), "Wrong class for panels in panel list"
        assert (
            len(panel_list) == 120
        ), "Panel list for the ALMA DA is produced with the wrong number of panels"
        assert (
            panel_list[0].label == "2-11"
        ), "Labelling for ALMA DA panels is not working as expected"
        assert (
            panel_list[72].label == "6-44"
        ), "Labelling for ALMA DA panels is not working as expected"
        assert (
            panel_list[-1].label == "3-51"
        ), "Labelling for ALMA DA panels is not working as expected"

        tel = get_proper_telescope("alma", "tp4")
        panel_list = tel.build_panel_list("flexible", 0.2)
        assert isinstance(
            panel_list[0], RingPanel
        ), "Wrong class for panels in panel list"
        assert (
            len(panel_list) == 205
        ), "Panel list for the ALMA TP is produced with the wrong number of panels"
        assert (
            panel_list[0].label == "1-11"
        ), "Labelling for ALMA TP panels is not working as expected"
        assert (
            panel_list[72].label == "3-45"
        ), "Labelling for ALMA TP panels is not working as expected"
        assert (
            panel_list[-1].label == "2-71"
        ), "Labelling for ALMA TP panels is not working as expected"

        return

    def test_build_ngvla_panel_list(self):
        model = "flexible"
        margin = 0.2
        tel = get_proper_telescope("ngvla")
        panel_list = tel.build_panel_list(model, margin)

        assert len(panel_list) == len(
            tel.panel_dict.keys()
        ), "Panel list has the wrong number of elements"
        assert (
            panel_list[0].model_name == model
        ), "Panels have initialized with the wrong model"
        assert (
            panel_list[0].margin == margin
        ), "Panels have initialized with the wrong margin"

    def test_assign_ringed_panel(self):
        tel = get_proper_telescope("vla")
        u_axis = simple_axis([-15, 15], 0.1, 0.0)
        v_axis = simple_axis([-15, 15], 0.1, 0.0)
        panel_list = tel.build_panel_list("flexible", 0.2)
        mask, radius, phi = tel.create_aperture_mask(
            u_axis, v_axis, return_polar_meshes=True
        )
        dev = np.where(mask, 0.0, np.nan)
        panel_map = tel.attribute_pixels_to_panels(
            panel_list, u_axis, v_axis, radius, phi, dev, mask
        )
        assert (
            panel_map.shape[0] == u_axis.shape[0]
            and panel_map.shape[1] == v_axis.shape[0]
        ), "panel map has the wrong shape"

        assert np.isnan(
            panel_map[u_axis.shape[0] // 2, v_axis.shape[0] // 2]
        ), "Panel map has a valid value for blocked region in aperture"
        assert np.isnan(panel_map[0, 0]), "Panel map has a valid value outside aperture"
        assert panel_map[220, 150] == 39.0, "Wrong panel assignment"
        assert panel_map[100, 150] == 12.0, "Wrong panel assignment"

        return

    def test_assign_ngvla_panel(self):
        tel = get_proper_telescope("ngvla")
        u_axis = simple_axis([-10, 10], 0.1, 0.0)
        v_axis = simple_axis([-10, 10], 0.1, 0.0)
        panel_list = tel.build_panel_list("flexible", 0.2)
        mask, radius, phi = tel.create_aperture_mask(
            u_axis, v_axis, return_polar_meshes=True
        )
        dev = np.where(mask, 0.0, np.nan)
        panel_map = tel.attribute_pixels_to_panels(
            panel_list, u_axis, v_axis, radius, phi, dev, mask
        )
        assert (
            panel_map.shape[0] == u_axis.shape[0]
            and panel_map.shape[1] == v_axis.shape[0]
        ), "panel map has the wrong shape"

        assert np.isnan(panel_map[0, 0]), "Panel map has a valid value outside aperture"
        assert panel_map[100, 120] == 40.0, "Wrong panel assignment"
        assert panel_map[50, 150] == 66.0, "Wrong panel assignment"
