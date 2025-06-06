from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.antenna.telescope import get_proper_telescope
from astrohack import extract_holog, extract_pointing, holog
from astrohack.utils.conversion import convert_unit

import numpy as np
import toolviper
import shutil
import xarray as xr


datafolder = "paneldata/"


def setup():
    # Download relevant panel test files
    toolviper.utils.data.download(
        file="ea25_cal_small_after_fixed.split.ms", folder=datafolder
    )

    extract_pointing(
        ms_name=f"{datafolder}/ea25_cal_small_after_fixed.split.ms",
        point_name=f"{datafolder}/ea25_cal_small_after_fixed.split.point.zarr",
        overwrite=True,
        parallel=False,
    )

    # Extract holography data using holog_obd_dict

    extract_holog(
        ms_name=f"{datafolder}/ea25_cal_small_after_fixed.split.ms",
        point_name=f"{datafolder}/ea25_cal_small_after_fixed.split.point.zarr",
        holog_name=f"{datafolder}/ea25_cal_small_after_fixed.split.holog.zarr",
        data_column="CORRECTED_DATA",
        ddi=0,
        parallel=False,
        overwrite=True,
    )

    holog(
        holog_name=f"{datafolder}/ea25_cal_small_after_fixed.split.holog.zarr",
        image_name=f"{datafolder}/ea25_cal_small_after_fixed.split.image.zarr",
        ant="ea25",
        overwrite=True,
        parallel=False,
    )


def cleanup():
    shutil.rmtree(datafolder)


class TestClassAntennaSurface:
    setup()
    ampfits = datafolder + "amp.fits"
    devfits = datafolder + "dev.fits"
    inputxds = xr.open_zarr(
        f"{datafolder}/ea25_cal_small_after_fixed.split.image.zarr/ant_ea25/ddi_0"
    )
    inputxds.attrs["ant_name"] = "test"
    inputxds.attrs["ddi"] = "test"
    tel = get_proper_telescope("vla")
    datashape = (510, 510)
    middlepix = 255
    tant = AntennaSurface(inputxds, tel, panel_margins=0.2)
    tolerance = 1e-6
    sigma = 20
    rand = sigma * np.random.randn(*datashape)
    zero = np.zeros(datashape)

    def test_init(self):
        """
        Tests the initialization of a AntennaSurface object
        """
        assert np.isnan(self.tant.in_rms), "RMS is not properly initialized"
        assert np.isnan(self.tant.ingains), "Gains are not properly initialized"
        assert self.tant.telescope.ringed, "Currently only ringed telescopes supported"
        assert self.tant.panelmodel == "rigid", "Default panel kind should be rigid"
        # Tests _build_polar
        assert (
            self.tant.rad.shape == self.datashape
        ), "Radius image does not have the expected dimensions"
        assert abs(self.tant.rad[self.middlepix, self.middlepix]) < 15e-1, (
            "Radius at the center of the image " "is more than 15 cm from zero"
        )
        assert (
            abs(
                self.tant.phi[self.middlepix, int(3 * self.datashape[0] / 4)]
                - np.pi / 2
            )
            / np.pi
            < 0.01
        ), "Azimuth at the horizontal axis is more than 1% different from pi/2"
        # tests _build_ring_panels
        assert len(self.tant.panels) == np.sum(self.tant.telescope.npanel), (
            "Number of panels do not " "match the expected number"
        )
        # tests _build_ring_mask
        assert (
            self.tant.mask.shape == self.datashape
        ), "Mask image does not have the expected dimensions"
        assert not self.tant.mask[
            0, 0
        ], "Mask is True at edges, where it should be False"

    def test_compile_panel_points_ringed(self):
        """
        Tests that a point falls into the correct panel and that this panel has the correct number of samples
        """
        compvaluep0 = [
            3.3030790441176467,
            0.43083639705882354,
            197,
            262,
            2.57549549e-04,
        ]
        compnsampp0 = 179
        self.tant.compile_panel_points()

        assert (
            len(self.tant.panels[0].samples) == compnsampp0
        ), "Number of samples in panel is different from reference"

        assert np.allclose(
            self.tant.panels[0].samples[0].get_array(), compvaluep0, atol=self.tolerance
        ), (
            "Point data in Panel is different from what is expected. Given values: "
            + str(self.tant.panels[0].samples[0].get_array())
            + " Expected values: "
            + str(compvaluep0)
        )

    def test_fit_surface(self):
        """
        Tests that fitting results for two panels match the reference
        """
        solveparsp0 = [0.00032415, 0.00037302, -0.00092434]
        solveparsp30 = [0.00038105, -0.00039928, -0.00067004]
        self.tant.fit_surface()

        assert len(self.tant.panels[0].model.parameters) == len(solveparsp0), (
            "Fitted results have a different length" " from reference"
        )

        for i in range(len(solveparsp30)):
            assert (
                abs(self.tant.panels[0].model.parameters[i] - solveparsp0[i])
                < self.tolerance
            ), "Fitting results for Panel 0 do not match reference within tolerance"
            assert (
                abs(self.tant.panels[30].model.parameters[i] - solveparsp30[i])
                < self.tolerance
            ), "Fitting results for Panel 30 do not match reference within tolerance"

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
        zgains = self.tant.gains()
        # If the antenna has not been corrected, gains returns a [2] list, if it has been corrected it returns a [2,2]
        # list containing the corrected gains. This try and except assures that this test works in both situations.
        try:
            len(zgains[0])
            assert zgains[0][0] == zgains[0][1]
        except TypeError:
            assert (
                zgains[0] == zgains[1]
            ), "Theoretical gains should be equal to real gains for a perfect antenna"
        self.tant.phase = self.rand
        rgains = self.tant.gains()
        assert (
            rgains[0] < rgains[1]
        ), "Real gains need to be inferior to theoretical gains on a noisy surface"

    def test_get_rms(self):
        """
        Tests RMS computations by using a zero array and a random array
        """
        self.tant.residuals = self.zero
        zrms = self.tant.get_rms()
        assert zrms[1] == 0, "RMS should be zero when computed over a zero array"
        self.tant.residuals = self.rand
        self.tant.mask[:, :] = True
        fac = convert_unit("mm", "m", "length")
        rrms = self.tant.get_rms()[1] * fac
        assert (
            abs(rrms - self.sigma) / self.sigma < 0.01
        ), "Computed RMS does not match expected RMS within 1%"

    def test_cleanup(self):
        cleanup()
