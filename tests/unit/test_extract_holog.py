import pytest

import os
import json
import shutil
import astrohack

from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing
from astrohack.extract_holog import generate_holog_obs_dict

class TestExtractHolog():
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given test class
        such as fetching test data """
        astrohack.gdown_utils.download(file="ea25_cal_small_after_fixed.split.ms", folder="data/", unpack=True)

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

    def test_extract_holog_obs_dict(self):

        # Generate pointing file
        extract_pointing(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            overwrite=True,
            parallel=False
        )

        # Generate a holog observations dictionary with a subset of data descirbed by ddi=1
        holog_obs_dict = generate_holog_obs_dict(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            ddi=[1],
            baseline_average_distance='all',
            baseline_average_nearest='all',
            parallel=False
        )

        # Extract holography data using holog_obd_dict
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            holog_obs_dict=holog_obs_dict,
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )

        # Get holog_obs_dict created by extract_holog
        with open(".holog_obs_dict.json") as holog_dict_file:
            holog_obs_test_dict = json.load(holog_dict_file)
            
        holog_obs_test_dict = json.loads(holog_obs_test_dict)

        # Check that the holog_obs_dict used in extract_holog matches the input holog_obs_dict
        assert holog_obs_test_dict == holog_obs_dict

    def test_extract_holog_ddi(self):

        # Generate pointing file
        extract_pointing(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            overwrite=True,
            parallel=False
        )

        # Generate a holog observations dictionary with a subset of data descirbed by ddi=1
        holog_obs_dict = generate_holog_obs_dict(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            ddi=[1],
            baseline_average_distance='all',
            baseline_average_nearest='all',
            parallel=False
        )

        # Extract holography data using holog_obd_dict
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            ddi=[1],
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )
        

        # Check that the holog_obs_dict used in extract_holog matches the input holog_obs_dict
        assert list(holog_mds.keys()) == ['ddi_1']

    def test_extract_holog_overwrite(self):

        # Generate pointing file
        extract_pointing(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            overwrite=True,
            parallel=False
        )

        # Generate a holog observations dictionary with a subset of data descirbed by ddi=1
        holog_obs_dict = generate_holog_obs_dict(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            baseline_average_distance='all',
            baseline_average_nearest='all',
            parallel=False
        )

        # Extract holography data
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )
        
        initial_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.holog.zarr')

        # Extract holography data
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )

        final_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.holog.zarr')

        # Check that the holog file date has change
        assert initial_time != final_time

    def test_extract_holog_baseline_average_distance(self):

        # extract pointing data
        pnt_mds = extract_pointing(
            ms_name='data/ea25_cal_small_after_fixed.split.ms',
            point_name='data/ea25_cal_small_after_fixed.split.point.zarr',
            parallel=False,
            overwrite=True
        )

        # Extract holography data
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            baseline_average_distance=100.1,
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )
        
        # Check that the expected antenna is present.
        assert list(holog_mds['ddi_0']['map_0'].keys()) == ['ant_ea25']

    def test_extract_holog_baseline_average_nearest(self):

        # extract pointing data
        pnt_mds = extract_pointing(
            ms_name='data/ea25_cal_small_after_fixed.split.ms',
            point_name='data/ea25_cal_small_after_fixed.split.point.zarr',
            parallel=False,
            overwrite=True
        )

        # Extract holography data
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            baseline_average_nearest=1,
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )
        
        # Check that the expected antenna is present.
        assert list(holog_mds['ddi_0']['map_0'].keys()).sort() == ['ant_ea25', 'ant_ea06'].sort()