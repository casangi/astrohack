import shutil

import astrohack

from astrohack import holog
from astrohack.dio import open_holog
from astrohack.dio import open_image
from astrohack.dio import open_panel
from astrohack.dio import open_pointing

from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing

from astrohack.panel import panel


class TestAstrohackDio():
    datafolder = 'dioData'
    holog_mds = dict()
    image_mds = dict()
    panel_mds = dict()

    @classmethod
    def setup_class(cls):
        astrohack.data.datasets.download(file="ea25_cal_small_after_fixed.split.ms", folder=cls.datafolder)

        extract_pointing(
            ms_name=cls.datafolder + '/ea25_cal_small_after_fixed.split.ms',
            point_name=cls.datafolder + '/ea25_cal_small_after_fixed.split.point.zarr',
            parallel=True,
            overwrite=True
        )

        cls.holog_mds = extract_holog(
            ms_name=cls.datafolder + '/ea25_cal_small_after_fixed.split.ms',
            point_name=cls.datafolder + '/ea25_cal_small_after_fixed.split.point.zarr',
            data_column='CORRECTED_DATA',
            parallel=True,
            overwrite=True
        )

        cls.image_mds = holog(
            holog_name=cls.datafolder + '/ea25_cal_small_after_fixed.split.holog.zarr',
            overwrite=True,
            phase_fit=True,
            apply_mask=True,
            to_stokes=True,
            parallel=True
        )

        cls.panel_mds = panel(
            image_name=cls.datafolder + '/ea25_cal_small_after_fixed.split.image.zarr',
            panel_model='rigid',
            parallel=True,
            overwrite=True
        )

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.datafolder)

    def test_open_holog(self):
        '''Open a holog file and return a holog data object'''
        holog_data = open_holog(self.datafolder + '/ea25_cal_small_after_fixed.split.holog.zarr')

        assert holog_data == self.holog_mds

    def test_open_image(self):
        '''Open an image file and return an image data object'''
        image_data = open_image(self.datafolder + '/ea25_cal_small_after_fixed.split.image.zarr')

        assert image_data == self.image_mds

    def test_open_panel(self):
        '''Open a panel file and return a panel data object'''
        panel_data = open_panel(self.datafolder + '/ea25_cal_small_after_fixed.split.panel.zarr')

        assert panel_data == self.panel_mds

    def test_open_pointing(self):
        '''Open a pointing file and return a pointing data object'''
        pointing_data = open_pointing(self.datafolder + '/ea25_cal_small_after_fixed.split.point.zarr')
        # check if keys match expected?
        # How to check xarray content...

        expected_keys = ['point_meta_ds', 'ant_ea25', 'ant_ea04', 'ant_ea06']

        for key in pointing_data.keys():
            assert key in expected_keys
