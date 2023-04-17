import pytest
from astrohack._classes.antenna_surface import AntennaSurface
from astrohack._classes.telescope import Telescope
from astrohack._utils._io import _aips_holog_to_xds
from astrohack._utils._conversion import _convert_unit

import numpy as np
import gdown
import shutil
import os


datafolder = "./paneldata/"


def setup():
    os.makedirs(name=datafolder, exist_ok=True)
    panelzip = datafolder + "panel.zip"
    if not os.path.exists(panelzip):
        url = "https://drive.google.com/u/1/uc?id=10fXyut_UHPUjIuaaEy6-m6wcycZHit2v&export=download"
        gdown.download(url, panelzip)
    shutil.unpack_archive(filename=panelzip, extract_dir=datafolder)


def cleanup():
    shutil.rmtree(datafolder)


class TestClassAntennaSurface:
    setup()
    ampfits = datafolder+'amp.fits'
    devfits = datafolder+'dev.fits'
    inputxds = _aips_holog_to_xds(ampfits, devfits)
    inputxds.attrs['ant_name'] = 'test'
    inputxds.attrs['ddi'] = 'test'
    tel = Telescope('vla')
    datashape = (256, 256)
    middlepix = 128
    tant = AntennaSurface(inputxds, tel)
    tolerance = 1e-6
    sigma = 20
    rand = sigma * np.random.randn(*datashape)
    zero = np.zeros(datashape)

    def test_init(self):
        """
        Tests the initialization of a AntennaSurface object
        """
        assert np.isnan(self.tant.in_rms), 'RMS is not properly initialized'
        assert np.isnan(self.tant.ingains), 'Gains are not properly initialized'
        assert self.tant.telescope.ringed, 'Currently only ringed telescopes supported'
        assert self.tant.panelmodel == 'rigid', 'Default panel kind should be rigid'
        # Tests _build_polar
        assert self.tant.rad.shape == self.datashape, 'Radius image does not have the expected dimensions'
        assert abs(self.tant.rad[self.middlepix, self.middlepix]) < 15e-1, 'Radius at the center of the image ' \
                                                                           'is more than 15 cm from zero'
        assert abs(self.tant.phi[self.middlepix, int(3 * self.datashape[0] / 4)] - np.pi / 2) / np.pi < 0.01, \
            'Azimuth at the horizontal axis is more than 1% different from pi/2'
        # tests _build_ring_panels
        assert len(self.tant.panels) == np.sum(self.tant.telescope.npanel), 'Number of panels do not ' \
                                                                            'match the expected number'
        # tests _build_ring_mask
        assert self.tant.mask.shape == self.datashape, 'Mask image does not have the expected dimensions'
        assert not self.tant.mask[0, 0], 'Mask is True at edges, where it should be False'

    def test_compile_panel_points_ringed(self):
        """
        Tests that a point falls into the correct panel and that this panel has the correct number of samples
        """
        compvaluep0 = [2.2265625, 0.703125, 147, 135, 0.0003463581378952494]
        compnsampp0 = 67
        self.tant.compile_panel_points()
        assert len(self.tant.panels[0].samples) == compnsampp0, 'Number of samples in panel is different from reference'
        assert self.tant.panels[0].samples[0] == compvaluep0, 'Point data in Panel is different from what is ' \
                                                              'expected'

    def test_fit_surface(self):
        """
        Tests that fitting results for two panels match the reference
        """
        solveparsp0 = [0.00024335, 0.00025452, -0.00035676]
        solveparsp30 = [0.00074635, -0.00059127, -0.00185721]
        self.tant.fit_surface()
        assert len(self.tant.panels[0].par) == len(solveparsp0), 'Fitted results have a different length from reference'
        for i in range(len(solveparsp30)):
            assert abs(self.tant.panels[0].par[i] - solveparsp0[i]) < self.tolerance, \
                'Fitting results for Panel 0 do not match reference within tolerance'
            assert abs(self.tant.panels[30].par[i] - solveparsp30[i]) < self.tolerance, \
                'Fitting results for Panel 30 do not match reference within tolerance'

    def test_correct_surface(self):
        """
        Tests that surface corrections and residuals combined properly reconstruct the original deviations
        """
        self.tant.correct_surface()
        reconstruction = self.tant.residuals - self.tant.corrections
        assert np.nansum((reconstruction - self.tant.deviation)[self.tant.mask]) < self.tolerance, \
            'Reconstruction is not faithful to original data'

    def test_gains_array(self):
        """
        Tests gain computations by using a zero array and a random array
        """
        zgains = self.tant._gains_array(self.zero)
        assert zgains[0] == zgains[1], 'Theoretical gains should be equal to real gains for a perfect antenna'
        rgains = self.tant._gains_array(self.rand)
        assert rgains[0] < rgains[1], 'Real gains need to be inferior to theoretical gains on a noisy surface'

    def test_get_rms(self):
        """
        Tests RMS computations by using a zero array and a random array
        """
        self.tant.residuals = self.zero
        zrms = self.tant.get_rms()
        assert zrms[1] == 0, 'RMS should be zero when computed over a zero array'
        self.tant.residuals = self.rand
        self.tant.mask[:, :] = True
        fac = _convert_unit('mm', 'm', 'length')
        rrms = self.tant.get_rms()[1]*fac
        assert abs(rrms - self.sigma)/self.sigma < 0.01, 'Computed RMS does not match expected RMS within 1%'

    def test_cleanup(self):
        cleanup()