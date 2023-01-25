import pytest
from astrohack._classes.antenna_surface import AntennaSurface
import numpy as np
import gdown
import shutil
import os


datafolder = "./paneldata/"


@pytest.fixture(scope="session", autouse=True)
def data_download_and_cleanup():
    os.makedirs(name=datafolder, exist_ok=True)
    panelzip = datafolder + "panel.zip"
    if not os.path.exists(panelzip):
        url = "https://drive.google.com/u/1/uc?id=10fXyut_UHPUjIuaaEy6-m6wcycZHit2v&export=download"
        gdown.download(url, panelzip)
    shutil.unpack_archive(filename=panelzip, extract_dir=datafolder)

    yield

    shutil.rmtree(datafolder)


class TestClassAntennaSurface:
    ampfits = datafolder+'amp.fits'
    devfits = datafolder+'dev.fits'
    datashape = (256, 256)
    middlepix = 128
    testantenna = AntennaSurface(ampfits, devfits, 'vla')
    tolerance = 1e-6
    sigma = 20
    rand = sigma * np.random.randn(*datashape)
    zero = np.zeros(datashape)

    def test_init(self):
        """
        Tests the initialization of a AntennaSurface object
        """
        assert np.isnan(self.testantenna.inrms), 'RMS is not properly initialized'
        assert np.isnan(self.testantenna.ingains), 'Gains are not properly initialized'
        assert self.testantenna.telescope.ringed, 'Currently only ringed telescopes supported'
        assert self.testantenna.panelkind == 'fixedtheta', 'Default panel kind should be fixedtheta'
        # test _read_images
        assert self.testantenna.amp.shape == self.datashape, 'Amplitude image does not have the expected dimensions'
        assert self.testantenna.dev.shape == self.datashape, 'Deviation image does not have the expected dimensions'
        # Tests _build_polar
        assert self.testantenna.rad.shape == self.datashape, 'Radius image does not have the expected dimensions'
        assert abs(self.testantenna.rad[self.middlepix, self.middlepix]) < 1e-1, 'Radius at the center of the image ' \
                                                                                 'is more than 10 cm from zero'
        assert abs(self.testantenna.phi[self.middlepix, int(3*self.datashape[0]/4)] - np.pi/2)/np.pi < 0.01, \
            'Azimuth at the horizontal axis is more than 1% different from pi/2'
        # tests _build_ring_panels
        assert len(self.testantenna.panels) == np.sum(self.testantenna.telescope.npanel), 'Number of panels do not ' \
                                                                                          'match the expected number'
        assert self.testantenna.panels[12].iring == 2, 'Ring numbering is different from what is expected'
        assert self.testantenna.panels[12].ipanel == 1, 'Panel numbering is different from what is expected'
        # tests _build_ring_mask
        assert self.testantenna.mask.shape == self.datashape, 'Mask image does not have the expected dimensions'
        assert not self.testantenna.mask[0, 0], 'Mask is True at edges, where it should be False'

    def test_compile_panel_points_ringed(self):
        """
        Tests that a point falls into the correct panel and that this panel has the correct number of samples
        """
        compvaluep0 = [2.51953125, 0.05859375, 149, 129, -0.30232558397962916]
        compnsampp0 = 164
        self.testantenna.compile_panel_points()
        assert self.testantenna.panels[0].nsamp == compnsampp0, 'Number of samples in panel is different from reference'
        assert self.testantenna.panels[0].values[0] == compvaluep0, 'Point data in Panel is different from what is ' \
                                                                    'expected'

    def test_fit_surface(self):
        """
        Tests that fitting results for two panels match the reference
        """
        solveparsp0 = [2.89665401e+04, 2.71826132e+00, 6.58578734e-01]
        solveparsp30 = [8.25412096e+03,  7.73518788e+03, -4.51138701e-01]
        self.testantenna.fit_surface()
        assert len(self.testantenna.panels[0].par) == len(solveparsp0), 'Fitted results have a different length from ' \
                                                                        'reference'
        for i in range(len(solveparsp30)):
            assert abs(self.testantenna.panels[0].par[i] - solveparsp0[i])/solveparsp0[i] < self.tolerance, \
                'Fitting results for Panel 0 do not match reference within tolerance'
            assert abs(self.testantenna.panels[30].par[i] - solveparsp30[i])/solveparsp30[i] < self.tolerance, \
                'Fitting results for Panel 30 do not match reference within tolerance'

    def test_correct_surface(self):
        """
        Tests that surface corrections and residuals combined properly reconstruct the original deviations
        """
        self.testantenna.correct_surface()
        reconstruction = self.testantenna.resi-self.testantenna.corr
        assert np.nansum((reconstruction-self.testantenna.dev)[self.testantenna.mask]) < self.tolerance, \
            'Reconstruction is not faithful to original data'

    def test_export_screw_adjustments(self):
        """
        Tests that exported screw adjustmens are in agreement with reference
        """
        self.testantenna.export_screw_adjustments(datafolder+'test.txt')
        data = np.loadtxt(datafolder+'test.txt', skiprows=6, unpack=True)
        assert data[0][11] == 1, 'Ring numbering is wrong'
        assert data[1][11] == 12, 'Panel numbering is wrong'
        for i in range(2, 6):
            assert data[i][11] == 0.18, 'Screw adjustments do not match reference'

    def test_gains_array(self):
        """
        Tests gain computations by using a zero array and a random array
        """
        zgains = self.testantenna._gains_array(self.zero)
        assert zgains[0] == zgains[1], 'Theoretical gains should be equal to real gains for a perfect antenna'
        rgains = self.testantenna._gains_array(self.rand)
        assert rgains[0] < rgains[1], 'Real gains need to be inferior to theoretical gains on a noisy surface'

    def test_get_rms(self):
        """
        Tests RMS computations by using a zero array and a random array
        """
        self.testantenna.resi = self.zero
        zrms = self.testantenna.get_rms()
        assert zrms[1] == 0, 'RMS should be zero when computed over a zero array'
        self.testantenna.resi = self.rand
        self.testantenna.mask[:, :] = True
        rrms = self.testantenna.get_rms()
        assert abs(rrms[1] - self.sigma)/self.sigma < 0.01, 'Computed RMS does not match expected RMS within 1%'
