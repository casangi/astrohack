import pytest

import os
import json
import shutil
import astrohack

import numpy as np

from astrohack.holog import holog
from astrohack.panel import panel
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing
from astrohack.extract_holog import generate_holog_obs_dict

def relative_difference(result, expected):
        return 2*np.abs(result - expected)/(abs(result) + abs(expected))

class TestPanel():
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given test class
        such as fetching test data """
        astrohack.gdown_utils.download(file="ea25_cal_small_after_fixed.split.ms", folder="data/", unpack=True)
        
        astrohack.gdown_utils.download(file='extract_holog_verification.json')
        astrohack.gdown_utils.download(file='holog_numerical_verification.json')

        extract_pointing(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            overwrite=True,
            parallel=False
        )

        # Extract holography data using holog_obd_dict
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )

        extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            data_column='CORRECTED_DATA',
            parallel=False,
            overwrite=True
        )

        holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            overwrite=True,
            parallel=False
        )

        panel(
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr',
            panel_name='data/ea25_cal_small_after_fixed.split.panel.zarr',
            cutoff=0.2,
            panel_margins=0.2,
            panel_model='rigid',
            parallel=False,
            overwrite=True
        )

        with open('data/ea25_cal_small_after_fixed.split.image.zarr/.image_attr') as json_attr:
            cls.json_file = json.load(json_attr)

    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to setup_class
        such as deleting test data """
        shutil.rmtree("data")

    def setup_method(self):
        """ setup any state specific to all methods of the given class """
        pass

    def teardown_method(self):
        """ teardown any state that was previously setup for all methods of the given class """
        pass

    
    def test_panel_name(self):

        assert os.path.exists('data/ea25_cal_small_after_fixed.split.panel.zarr')

    def test_panel_ant_id(self):

        panel_mds = panel(
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr',
            panel_name='data/ea25_cal_small_after_fixed.split.panel.zarr',
            cutoff=0.2,
            panel_margins=0.2,
            ant_id=['ea25'],
            ddi=None,
            panel_model='rigid',
            parallel=False,
            overwrite=True
        )

        assert list(panel_mds.keys()) == ['ant_ea25']

    def test_panel_ddi(self):

        panel_mds = panel(
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr',
            panel_name='data/ea25_cal_small_after_fixed.split.panel.zarr',
            cutoff=0.2,
            panel_margins=0.2,
            ddi=[0],
            panel_model='rigid',
            parallel=False,
            overwrite=True
        )

        for ant in panel_mds.keys():
            for ddi in panel_mds[ant].keys():
                assert ddi == "ddi_0"

    def test_panel_overwrite(self):
        initial_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.image.zarr')
        
        panel_mds = panel(
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr',
            panel_name='data/ea25_cal_small_after_fixed.split.panel.zarr',
            cutoff=0.2,
            panel_margins=0.2,
            panel_model='rigid',
            parallel=False,
            overwrite=True
        )

        modified_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.panel.zarr')

        assert initial_time != modified_time

    def test_panel_not_overwrite(self):
        initial_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.panel.zarr')
        
        try:
            panel_mds = panel(
                image_name='data/ea25_cal_small_after_fixed.split.image.zarr',
                panel_name='data/ea25_cal_small_after_fixed.split.panel.zarr',
                cutoff=0.2,
                panel_margins=0.2,
                panel_model='rigid',
                parallel=False,
                overwrite=False
            )

        except FileExistsError:
            pass

        finally:
            modified_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.panel.zarr')

            assert initial_time == modified_time

        


    