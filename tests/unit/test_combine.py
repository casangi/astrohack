import os
import shutil
import graphviper

import numpy as np

from astrohack.holog import holog
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing
from astrohack.extract_holog import generate_holog_obs_dict
from astrohack.combine import combine

from graphviper.utils.data.remote import download


class TestCombine:
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given test class
        such as fetching test data """
        graphviper.utils.data.download(file="ea25_cal_small_before_fixed.split.ms", folder="data")

        # This gets the remote functionality for now
        download(file="combine_weight_array", folder="data")

        # Generate pointing file
        extract_pointing(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            overwrite=True,
            parallel=False
        )

        # Generate a holog observations dictionary with a subset of data described by ddi=1
        holog_obs_dict = generate_holog_obs_dict(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            baseline_average_distance='all',
            baseline_average_nearest='all',
            parallel=False
        )

        # Extract holography data using holog_obd_dict
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )

        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            overwrite=True,
            parallel=False
        )

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

    def test_combine_ddi(self):
        """
            Specify a ddi value to be process and check that it is the only one processed.
        """

        combine_mds = combine(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            combine_name="data/ea25_cal_small_before_fixed.split.combine.zarr",
            ant="all",
            ddi=[0],
            weighted=False,
            parallel=False,
            overwrite=True
        )

        assert list(combine_mds["ant_ea25"].keys()) == ["ddi_0"]

    def test_combine_ant(self):
        """
            Specify a ddi value to be process and check that it is the only one processed.
        """

        combine_mds = combine(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            combine_name="data/ea25_cal_small_before_fixed.split.combine.zarr",
            ant="ea25",
            ddi="all",
            weighted=False,
            parallel=False,
            overwrite=True
        )

        assert list(combine_mds.keys()) == ["ant_ea25"]

    def test_combine_weighted(self):
        """
            Specify a ddi value to be process and check that it is the only one processed.
        """

        combine_mds = combine(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            combine_name="data/ea25_cal_small_before_fixed.split.combine.weighted.zarr",
            ant="ea25",
            ddi="all",
            weighted=True,
            parallel=False,
            overwrite=True
        )

        with open("data/combine_weight.npy", "rb") as file:
            combine_weight_array = np.load(file)

        combine_mds_values = combine_mds["ant_ea25"]["ddi_0"].AMPLITUDE.values

        np.nan_to_num(combine_weight_array, copy=False)
        np.nan_to_num(combine_mds_values, copy=False)

        #assert (combine_weight_array == combine_mds_values).all()

    def test_combine_overwrite(self):
        """
            Specify that the output file should be overwritten if it exists; check that it is overwritten.
        """

        combine_mds = combine(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            combine_name="data/ea25_cal_small_before_fixed.split.combine.zarr",
            ant="all",
            ddi="all",
            weighted=False,
            parallel=False,
            overwrite=True
        )

        initial_time = os.path.getctime('data/ea25_cal_small_before_fixed.split.combine.zarr')

        # Combine image data
        combine_mds = combine(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            combine_name="data/ea25_cal_small_before_fixed.split.combine.zarr",
            ant="all",
            ddi="all",
            weighted=False,
            parallel=False,
            overwrite=True
        )

        final_time = os.path.getctime('data/ea25_cal_small_before_fixed.split.combine.zarr')

        # Check that the holog file date has changed
        assert initial_time != final_time