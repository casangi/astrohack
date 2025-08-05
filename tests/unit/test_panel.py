import os
import shutil
import pytest
import toolviper

import numpy as np

from astrohack.antenna import get_proper_telescope
from astrohack.holog import holog
from astrohack.panel import panel
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing


def relative_difference(result, expected):
    return 2 * np.abs(result - expected) / (abs(result) + abs(expected))


class TestPanel:
    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        toolviper.utils.data.download(
            file="ea25_cal_small_before_fixed.split.ms", folder="data/"
        )

        extract_pointing(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            overwrite=True,
            parallel=False,
        )

        # Extract holography data using holog_obd_dict

        extract_holog(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True,
        )

        holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            overwrite=True,
            parallel=False,
        )

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree("data")

    def setup_method(self):
        """setup any state specific to all methods of the given class"""
        pass

    def teardown_method(self):
        """teardown any state that was previously setup for all methods of the given class"""
        pass

    def test_panel_name(self):
        """
        Check that the panel output name was created correctly.
        """
        panel_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            parallel=False,
            overwrite=True,
        )
        assert os.path.exists("data/ea25_cal_small_before_fixed.split.panel.zarr")

    def test_panel_ant_id(self):
        """
        Specify a single antenna to process; check that only that antenna was processed.
        """

        panel_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            panel_name="data/ea25_cal_small_before_fixed.split.panel.zarr",
            clip_type="relative",
            clip_level=0.2,
            panel_margins=0.2,
            ant=["ea25"],
            panel_model="rigid",
            parallel=False,
            overwrite=True,
        )

        assert list(panel_mds.keys()) == ["ant_ea25"]

    def test_panel_ddi(self):
        """
        Specify a single ddi to process; check that only that ddi was processed.
        """

        panel_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            panel_name="data/ea25_cal_small_before_fixed.split.panel.zarr",
            clip_type="relative",
            clip_level=0.2,
            panel_margins=0.2,
            ddi=["0"],
            panel_model="rigid",
            parallel=False,
            overwrite=True,
        )

        for ant in panel_mds.keys():
            for ddi in panel_mds[ant].keys():
                assert ddi == "ddi_0"

    def test_panel_overwrite(self):
        """
        Specify the output file should be overwritten; check that it WAS.
        """
        initial_time = os.path.getctime(
            "data/ea25_cal_small_before_fixed.split.image.zarr"
        )

        panel_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            panel_name="data/ea25_cal_small_before_fixed.split.panel.zarr",
            clip_type="relative",
            clip_level=0.2,
            panel_margins=0.2,
            panel_model="rigid",
            parallel=False,
            overwrite=True,
        )

        modified_time = os.path.getctime(
            "data/ea25_cal_small_before_fixed.split.panel.zarr"
        )

        assert initial_time != modified_time

    def test_panel_not_overwrite(self):
        """
        Specify the output file should be NOT be overwritten; check that it WAS NOT.
        """
        initial_time = os.path.getctime(
            "data/ea25_cal_small_before_fixed.split.panel.zarr"
        )

        try:
            panel_mds = panel(
                image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
                panel_name="data/ea25_cal_small_before_fixed.split.panel.zarr",
                clip_type="relative",
                clip_level=0.2,
                panel_margins=0.2,
                panel_model="rigid",
                parallel=False,
                overwrite=False,
            )

        except FileExistsError:
            pass

        finally:
            modified_time = os.path.getctime(
                "data/ea25_cal_small_before_fixed.split.panel.zarr"
            )

            assert initial_time == modified_time

    def test_panel_mode(self):
        """
        Specify panel computation mode and check that the data rms responded as expected.
        """
        panel_list = ["3-4", "5-27", "5-37", "5-38"]

        panel_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            panel_name="data/ea25_cal_small_before_fixed.split.panel.zarr",
            overwrite=True,
        )

        default_rms = (
            panel_mds["ant_ea25"]["ddi_0"]
            .sel(labels=panel_list)
            .map(np.std)
            .PANEL_SCREWS.values
        )

        panel_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            panel_name="data/ea25_cal_small_before_fixed.split.panel.zarr",
            panel_model="mean",
            overwrite=True,
        )

        mean_rms = (
            panel_mds["ant_ea25"]["ddi_0"]
            .sel(labels=panel_list)
            .map(np.std)
            .PANEL_SCREWS.values
        )

        assert mean_rms < default_rms

    def test_panel_absolute_clip(self):
        """
        Set cutoff=0 and compare results to known truth value array.
        """
        panel_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            clip_type="absolute",
            clip_level=0.0,
            use_detailed_mask=False,
            parallel=False,
            overwrite=True,
        )

        telescope = get_proper_telescope("vla")

        radius = panel_mds["ant_ea25"]["ddi_0"]["RADIUS"].values
        dish_mask = np.where(radius < telescope.outer_radial_limit, 1.0, 0)
        dish_mask = np.where(radius < telescope.inner_radial_limit, 0, dish_mask)
        nvalid_pix = np.sum(dish_mask)

        assert np.sum(panel_mds["ant_ea25"]["ddi_0"].MASK.values) == nvalid_pix

    def test_panel_relative_clip(self):
        panel_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            clip_type="relative",
            clip_level=1,
            parallel=False,
            overwrite=True,
        )

        assert np.sum(panel_mds["ant_ea25"]["ddi_0"].MASK.values) == 1

    def test_panel_sigma_clip(self):
        panel_sig2_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            panel_name="data/clip_sigma_2.split.panel.zarr",
            clip_type="sigma",
            clip_level=2,
            parallel=False,
            overwrite=True,
        )
        n_mask_sig2 = np.sum(panel_sig2_mds["ant_ea25"]["ddi_0"].MASK.values)

        panel_sig3_mds = panel(
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            panel_name="data/clip_sigma_3.split.panel.zarr",
            clip_type="sigma",
            clip_level=3,
            parallel=False,
            overwrite=True,
        )

        n_mask_sig3 = np.sum(panel_sig3_mds["ant_ea25"]["ddi_0"].MASK.values)

        assert n_mask_sig2 > n_mask_sig3
