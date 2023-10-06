import os
import json
import shutil
import astrohack

import numpy as np

from astrohack.holog import holog
from astrohack.panel import panel
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing


def relative_difference(result, expected):
    return 2 * np.abs(result - expected) / (abs(result) + abs(expected))


class TestPanel():
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given test class
        such as fetching test data """
        astrohack.data.datasets.download(file="ea25_cal_small_after_fixed.split.ms", folder="data/")

        astrohack.data.datasets.download(file='extract_holog_verification.json')
        astrohack.data.datasets.download(file='holog_numerical_verification.json')

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
        """
            Check that the panel output name was created correctly.
        """

        assert os.path.exists('data/ea25_cal_small_after_fixed.split.panel.zarr')

    def test_panel_ant_id(self):
        """
           Specify a single antenna to process; check that only that antenna was processed.
        """

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
        """
            Specify a single ddi to process; check that only that ddi was processed.
        """

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
        """
            Specify the output file should be overwritten; check that it WAS.
        """
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
        """
           Specify the output file should be NOT be overwritten; check that it WAS NOT.
        """
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

    def test_panel_mode(self):
        """
           Specify panel computation mode and check that the data rms responded as expected.
        """
        panel_list = ['3-4', '5-27', '5-37', '5-38']

        panel_mds = panel(
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr',
            panel_name='data/ea25_cal_small_after_fixed.split.panel.zarr',
            overwrite=True
        )

        default_rms = panel_mds["ant_ea25"]["ddi_0"].sel(labels=panel_list).apply(np.std).PANEL_SCREWS.values

        panel_mds = panel(
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr',
            panel_name='data/ea25_cal_small_after_fixed.split.panel.zarr',
            panel_model='mean',
            overwrite=True
        )

        mean_rms = panel_mds["ant_ea25"]["ddi_0"].sel(labels=panel_list).apply(np.std).PANEL_SCREWS.values

        assert mean_rms < default_rms

    def test_panel_cutoff(self):
        """
           Set cutoff=0 and compare results to known truth value array.
        """
        astrohack.data.datasets.download(file='panel_cutoff_mask')

        with open("panel_cutoff_mask.npy", "rb") as array:
            reference_array = np.load(array)

        panel_mds = panel(
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr',
            cutoff=0.0,
            parallel=False,
            overwrite=True
        )

        assert np.all(panel_mds["ant_ea25"]["ddi_0"].MASK.values == reference_array)
