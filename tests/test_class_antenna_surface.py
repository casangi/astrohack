import pytest
from astrohack._classes.antenna_surface import AntennaSurface
import numpy as np
import gdown
import shutil
import os


def download_test_data():
    datafolder = "./paneldata/"
    os.makedirs(name=datafolder, exist_ok=True)
    panelzip = datafolder + "panel.zip"
    if not os.path.exists(panelzip):
        url = "https://drive.google.com/u/1/uc?id=10fXyut_UHPUjIuaaEy6-m6wcycZHit2v&export=download"
        gdown.download(url, panelzip)
    shutil.unpack_archive(filename=panelzip, extract_dir=datafolder)
    return datafolder


class TestClassAntennaSurface:
    datafolder = download_test_data()
    ampfits = datafolder + "amp.fits"
    devfits = datafolder + "dev.fits"
    datashape = (256, 256)
    middlepix = 128
    testantenna = AntennaSurface(ampfits, devfits, "vla")

    def test_init(self):
        assert np.isnan(self.testantenna.inrms)
        assert np.isnan(self.testantenna.ingains)
        assert self.testantenna.telescope.ringed
        assert self.testantenna.panelkind == "fixedtheta"
        # test _read_images
        assert self.testantenna.amp.shape == self.datashape
        assert self.testantenna.dev.shape == self.datashape
        # Tests _build_polar
        assert self.testantenna.rad.shape == self.datashape
        assert abs(self.testantenna.rad[self.middlepix, self.middlepix]) < 1e-1
        assert (
            abs(
                self.testantenna.phi[self.middlepix, int(3 * self.datashape[0] / 4)]
                - np.pi / 2
            )
            < 1e-2
        )
        # tests _build_ring_panels
        assert len(self.testantenna.panels) == np.sum(self.testantenna.telescope.npanel)
        assert self.testantenna.panels[12].iring == 2
        assert self.testantenna.panels[12].ipanel == 1
        # tests _build_ring_mask
        assert self.testantenna.mask.shape == self.datashape
        assert not self.testantenna.mask[0, 0]

    def test_compile_panel_points_ringed(self):
        compvaluep0 = [2.51953125, 0.05859375, 149, 129, -0.30232558397962916]
        compnsampp0 = 164
        self.testantenna.compile_panel_points()
        assert self.testantenna.panels[0].nsamp == compnsampp0
        assert self.testantenna.panels[0].values[0] == compvaluep0

    def test_fit_surface(self):
        solveparsp0 = [2.89665401e04, 2.71826132e00, 6.58578734e-01]
        solveparsp30 = [8.25412096e03, 7.73518788e03, -4.51138701e-01]
        self.testantenna.fit_surface()
        assert len(self.testantenna.panels[0].par) == len(solveparsp0)
        for i in range(len(solveparsp30)):
            assert abs(self.testantenna.panels[0].par[i] - solveparsp0[i]) < 1e-4
            assert abs(self.testantenna.panels[30].par[i] - solveparsp30[i]) < 1e-4
