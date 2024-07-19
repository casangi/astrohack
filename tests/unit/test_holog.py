import pytest

import os
import json
import shutil
import graphviper

import numpy as np

from astrohack.holog import holog
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing


def relative_difference(result, expected):
    return 2 * np.abs(result - expected) / (abs(result) + abs(expected))


class TestHolog():
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given test class
        such as fetching test data """
        graphviper.utils.data.download(file="ea25_cal_small_before_fixed.split.ms", folder="data/")

        graphviper.utils.data.download(file='holog_numerical_verification.json', folder="data/")

        extract_pointing(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            overwrite=True,
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

        extract_holog(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            data_column='CORRECTED_DATA',
            parallel=False,
            overwrite=True
        )

        holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr') as json_attr:
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

    def test_holog_grid_cell_size(self):
        """
           Calculate the correct grid and cell size when compared to known values in the test file; known values are
           provided by a test json file.
        """

        tolerance = 2.e-5

        with open("data/holog_numerical_verification.json") as file:
            reference_dict = json.load(file)

        with open('data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr') as attr_file:
            image_attr = json.load(attr_file)

        for i, _ in enumerate(image_attr['cell_size']):
            print(image_attr['grid_size'][i])
            print(reference_dict["vla"]['grid_size'][i])

            assert relative_difference(
                image_attr['cell_size'][i],
                reference_dict["vla"]['cell_size'][i]
            ) < tolerance

            assert relative_difference(
                image_attr['grid_size'][i],
                reference_dict["vla"]['grid_size'][i]
            ) < tolerance

    def test_holog_image_name(self):
        """
            Test holog image name created correctly.
        """

        assert os.path.exists('data/ea25_cal_small_before_fixed.split.image.zarr')

    def test_holog_ant_id(self):
        """
            Specify a single antenna to process; check that is the only antenna returned.
        """

        image_mds = holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            ant=['ea25'],
            overwrite=True,
            parallel=False
        )

        assert list(image_mds.keys()) == ['ant_ea25']

    def test_holog_ddi(self):
        """
            Specify a single ddi to process; check that is the only ddi returned.
        """

        image_mds = holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            overwrite=True,
            ddi=[0],
            parallel=False
        )

        for ant in image_mds.keys():
            for ddi in image_mds[ant].keys():
                assert ddi == "ddi_0"

    def test_holog_padding_factor(self):
        """
            Specify a padding factor to use in the image creation; check that image size is created.
        """

        image_mds = holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            padding_factor=50,
            overwrite=True,
            parallel=False
        )

        for ant in image_mds.keys():
            for ddi in image_mds[ant].keys():
                assert image_mds[ant][ddi].APERTURE.shape == (1, 1, 4, 512, 512)

    def test_holog_chan_average(self):
        """
            Check that channel average flag was set holog is run.
        """
        image_mds = holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            chan_average=True,
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)

        assert json_file['chan_average'] is True

    def test_holog_scan_average(self):
        """
            Check that scan average flag was set holog is run.
        """
        image_mds = holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            scan_average=False,
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)

        assert json_file['scan_average'] == False

    def test_holog_grid_interpolation(self):
        """
            Check that grid interpolation flag was set holog is run.
        """
        image_mds = holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            grid_interpolation_mode='nearest',
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)

        assert json_file['grid_interpolation_mode'] == 'nearest'

    def test_holog_chan_tolerance(self):
        """
            Check that channel tolerance is propagated correctly.
        """
        image_mds = holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            chan_tolerance_factor=0.0049,
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)

        assert json_file['chan_tolerance_factor'] == 0.0049

    def test_hologto_stokes(self):
        """
            Check that to_stokes flag was set holog is run.
        """
        image_mds = holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            to_stokes=True,
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)

        assert json_file['to_stokes'] == True

        assert (image_mds['ant_ea25']['ddi_0'].pol.values == np.array(['I', 'Q', 'U', 'V'])).all()

    def test_holog_overwrite(self):
        """
            Specify the output file should be overwritten; check that it WAS.
        """
        initial_time = os.path.getctime('data/ea25_cal_small_before_fixed.split.image.zarr')

        image_mds = holog(
            holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
            overwrite=True,
            parallel=False
        )

        modified_time = os.path.getctime('data/ea25_cal_small_before_fixed.split.image.zarr')

        assert initial_time != modified_time

    def test_holog_not_overwrite(self):
        """
           Specify the output file should be NOT be overwritten; check that it WAS NOT.
        """
        initial_time = os.path.getctime('data/ea25_cal_small_before_fixed.split.image.zarr')

        try:
            holog(
                holog_name='data/ea25_cal_small_before_fixed.split.holog.zarr',
                image_name='data/ea25_cal_small_before_fixed.split.image.zarr',
                overwrite=False,
                parallel=False
            )

        except FileExistsError:
            pass

        finally:
            modified_time = os.path.getctime('data/ea25_cal_small_before_fixed.split.image.zarr')

            assert initial_time == modified_time
