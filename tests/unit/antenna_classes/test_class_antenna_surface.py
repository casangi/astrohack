import json
import shutil

import numpy as np
import toolviper

from astrohack import open_image
from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.utils.conversion import convert_unit
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    execute_cleanup,
    produce_reference_data,
)

datafolder = "paneldata/"


class TestClassAntennaSurface:
    data_dir = "ant_class_data"
    img_name = "ea25_cal_before_reference.image.zarr"

    ant_key = "ant_ea25"
    ddi_key = "ddi_0"

    datashape = (510, 510)
    middle_pix = 255
    tolerance = 1e-6
    sigma = 20
    rand = sigma * np.random.randn(*datashape)
    zero = np.zeros(datashape)

    json_file_name = "ant_class_ref.json"
    ref_json_dict = {}

    @classmethod
    def setup_class(cls):
        toolviper.utils.data.download(cls.img_name, cls.data_dir)
        toolviper.utils.data.download(cls.json_file_name, cls.data_dir)

        add_data_folder_to_names_in_class(cls)

        input_xds = open_image(cls.img_name)[cls.ant_key][cls.ddi_key].dataset
        input_xds.attrs["ant_name"] = "ea00"
        input_xds.attrs["ddi"] = "test"

        cls.tant = AntennaSurface(input_xds, panel_margins=0.2)

        return

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if produce_reference_data():
            with open(cls.json_file_name, "w") as json_file:
                json.dump(cls.ref_json_dict, json_file)
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)

    def test_init(self):
        """
        Tests the initialization of a AntennaSurface object
        """
        assert np.isnan(self.tant.in_rms), "RMS is not properly initialized"
        assert np.isnan(self.tant.ingains), "Gains are not properly initialized"
        assert self.tant.telescope.ringed, "Currently only ringed telescopes supported"
        assert self.tant.panelmodel == "rigid", "Default panel kind should be rigid"
        # Tests _build_polar
        assert self.tant.rad.shape == self.datashape, (
            "Radius image does not have the expected dimensions"
        )
        assert abs(self.tant.rad[self.middle_pix, self.middle_pix]) < 15e-1, (
            "Radius at the center of the image is more than 15 cm from zero"
        )
        assert (
            abs(
                self.tant.phi[self.middle_pix, int(3 * self.datashape[0] / 4)]
                - np.pi / 2
            )
            / np.pi
            < 0.01
        ), "Azimuth at the horizontal axis is more than 1% different from pi/2"
        # tests _build_ring_panels
        assert len(self.tant.panels) == np.sum(self.tant.telescope.n_panel_per_ring), (
            "Number of panels do not match the expected number"
        )
        # tests _build_ring_mask
        assert self.tant.mask.shape == self.datashape, (
            "Mask image does not have the expected dimensions"
        )
        assert not self.tant.mask[0, 0], (
            "Mask is True at edges, where it should be False"
        )

    def test_fit_surface(self):
        """
        Tests that fitting results for two panels match the reference
        """
        expected_len = 3
        self.tant.fit_surface()
        panel_fit_res = []
        for panel in self.tant.panels:
            panel_fit_res.append(panel.model.parameters.tolist())

        assert len(self.tant.panels[0].model.parameters) == expected_len, (
            "Fitted results have a different length from reference"
        )
        if produce_reference_data():
            self.ref_json_dict["solved_parameters"] = panel_fit_res
            return

        with open(self.json_file_name) as json_file:
            ref_json_dict = json.load(json_file)

        ref_solved_pars = ref_json_dict["solved_parameters"]

        assert len(ref_solved_pars) == len(self.tant.panels), (
            "Number of solved panels does not match with the number of panels in reference"
        )

        n_failed_panels = 0
        for i_panel, panel in enumerate(self.tant.panels):
            if not np.allclose(
                ref_solved_pars[i_panel], panel.model.parameters, atol=self.tolerance
            ):
                n_failed_panels += 1
        assert n_failed_panels == 0, (
            f"Fitting results differ for {n_failed_panels} panels."
        )

    def test_correct_surface(self):
        """
        Tests that surface corrections and residuals combined properly reconstruct the original deviations
        """
        self.tant.correct_surface()
        reconstruction = self.tant.residuals - self.tant.corrections
        assert (
            np.nansum((reconstruction - self.tant.deviation)[self.tant.mask])
            < self.tolerance
        ), "Reconstruction is not faithful to original data"

    def test_gains_array(self):
        """
        Tests gain computations by using a zero array and a random array
        """
        self.tant.phase = self.zero
        z_gains = self.tant.gains()
        # If the antenna has not been corrected, gains returns a [2] list, if it has been corrected it returns a [2,2]
        # list containing the corrected gains. This try and except assures that this test works in both situations.
        try:
            len(z_gains[0])
            assert z_gains[0][0] == z_gains[0][1]
        except TypeError:
            assert z_gains[0] == z_gains[1], (
                "Theoretical gains should be equal to real gains for a perfect antenna"
            )
        self.tant.phase = self.rand
        r_gains = self.tant.gains()
        assert r_gains[0] < r_gains[1], (
            "Real gains need to be inferior to theoretical gains on a noisy surface"
        )

    def test_get_rms(self):
        """
        Tests RMS computations by using a zero array and a random array
        """
        self.tant.residuals = self.zero
        z_rms = self.tant.get_rms()
        assert z_rms[1] == 0, "RMS should be zero when computed over a zero array"
        self.tant.residuals = self.rand
        self.tant.mask[:, :] = True
        fac = convert_unit("mm", "m", "length")
        r_rms = self.tant.get_rms()[1] * fac
        assert abs(r_rms - self.sigma) / self.sigma < 0.01, (
            "Computed RMS does not match expected RMS within 1%"
        )
