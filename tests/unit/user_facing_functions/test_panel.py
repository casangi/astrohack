import os
import pathlib
import shutil

import numpy as np
import pytest
import toolviper

from astrohack import open_panel, panel
from astrohack.antenna.telescope import get_proper_telescope
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    execute_cleanup,
    produce_reference_data,
)


class TestPanel:
    data_dir = "panel_data"

    img_name = "ea25_cal_before_reference.image.zarr"

    def_pnl_name = "ea25_cal_before_reference.panel.zarr"
    ref_pnl_name = "ea25_before_reference.panel.zarr"

    ant_id = "ea25"
    ant_key = f"ant_{ant_id}"
    ddi_id = 0
    ddi_key = f"ddi_{ddi_id}"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        toolviper.utils.data.download(file=cls.img_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_pnl_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)

    def test_defaults(self):
        """
        Check that the panel output name was created correctly.
        """
        new_pnl_mds = panel(image_name=self.img_name, overwrite=True)
        assert pathlib.Path(self.def_pnl_name).is_dir(), (
            f"A .panel.zarr file named {self.def_pnl_name} does not exist."
        )

        if produce_reference_data():
            return
        ref_pnl_mds = open_panel(self.ref_pnl_name)
        assert new_pnl_mds.is_close_to(ref_pnl_mds), (
            "Reference and new mdses are different."
        )

    def test_data_selection(self):
        """
        Specify a single antenna to process; check that only that antenna was processed.
        """
        if produce_reference_data():
            return
        pnl_mds = panel(
            image_name=self.img_name,
            clip_type="relative",
            clip_level=0.2,
            panel_margins=0.2,
            ant=self.ant_id,
            ddi=self.ddi_id,
            panel_model="rigid",
            parallel=False,
            overwrite=True,
        )

        exp_ant_list = [self.ant_key]
        exp_ddi_list = [self.ddi_key]
        assert (list(pnl_mds.keys())) == exp_ant_list, (
            f"Expected {exp_ant_list} but got {list(pnl_mds.keys())}"
        )
        for ant_key, ant_xdt in pnl_mds.items():
            ddi_list = list(ant_xdt.keys())
            assert ddi_list == exp_ddi_list, (
                f"Expected {exp_ddi_list}, but got {ddi_list} for {ant_key}."
            )

    def test_overwrite(self):
        """
        Specify the output file should be overwritten; check that it WAS.
        """
        if produce_reference_data():
            return
        initial_time = os.path.getctime(self.def_pnl_name)

        panel(
            image_name=self.img_name,
            overwrite=True,
            ant=self.ant_id,
            ddi=self.ddi_id,
        )
        modified_time = os.path.getctime(self.def_pnl_name)
        assert initial_time != modified_time

        with pytest.raises(FileExistsError):
            panel(image_name=self.img_name, overwrite=False)

    def test_panel_model(self):
        """
        Specify panel computation mode and check that the data rms responded as expected.
        """
        if produce_reference_data():
            return
        panel_mds = panel(
            image_name=self.img_name,
            panel_name=self.def_pnl_name,
            overwrite=True,
            panel_model="flexible",
            ant=self.ant_id,
            ddi=self.ddi_id,
        )

        flexible_rms = np.std(
            panel_mds[self.ant_key][self.ddi_key]["PANEL_SCREWS"].values
        )

        panel_mds = panel(
            image_name=self.img_name,
            panel_name=self.def_pnl_name,
            panel_model="mean",
            overwrite=True,
            ant=self.ant_id,
            ddi=self.ddi_id,
        )

        mean_rms = np.std(panel_mds[self.ant_key][self.ddi_key]["PANEL_SCREWS"].values)

        assert mean_rms < flexible_rms, (
            f"Mean RMS ({mean_rms}) should be smaller than flexible RMS ({mean_rms}) as all screws for a panel have "
            f"the same value"
        )

    def test_absolute_clip(self):
        """
        Set cutoff=0 and compare results to known truth value array.
        """
        if produce_reference_data():
            return
        panel_mds = panel(
            image_name=self.img_name,
            clip_type="absolute",
            clip_level=0.0,
            use_detailed_mask=False,
            parallel=False,
            overwrite=True,
            ant=self.ant_id,
            ddi=self.ddi_id,
        )
        telescope = get_proper_telescope("vla")
        radius = panel_mds[self.ant_key][self.ddi_key]["RADIUS"].values
        dish_mask = np.where(radius < telescope.outer_radial_limit, 1.0, 0)
        dish_mask = np.where(radius < telescope.inner_radial_limit, 0, dish_mask)
        nvalid_pix = np.sum(dish_mask)
        assert (
            np.sum(panel_mds[self.ant_key][self.ddi_key].MASK.values) == nvalid_pix
        ), (
            "An absolute clip of level 0 should include all and only the pixels inside the aperture"
        )

    def test_relative_clip(self):
        if produce_reference_data():
            return
        panel_mds = panel(
            image_name=self.img_name,
            clip_type="relative",
            clip_level=1,
            parallel=False,
            overwrite=True,
            ant=self.ant_id,
            ddi=self.ddi_id,
        )

        assert np.sum(panel_mds[self.ant_key][self.ddi_key].MASK.values) == 1, (
            "A relative clip of level 1 should include only the brightest pixel in the aperture"
        )

    def test_sigma_clip(self):
        if produce_reference_data():
            return
        panel_sig2_mds = panel(
            image_name=self.img_name,
            clip_type="sigma",
            clip_level=2,
            parallel=False,
            overwrite=True,
            ant=self.ant_id,
            ddi=self.ddi_id,
        )
        n_mask_sig2 = np.sum(panel_sig2_mds[self.ant_key][self.ddi_key].MASK.values)

        panel_sig3_mds = panel(
            image_name=self.img_name,
            clip_type="sigma",
            clip_level=3,
            parallel=False,
            overwrite=True,
            ant=self.ant_id,
            ddi=self.ddi_id,
        )

        n_mask_sig3 = np.sum(panel_sig3_mds[self.ant_key][self.ddi_key].MASK.values)

        assert n_mask_sig2 > n_mask_sig3, (
            "A mask with clip at 2 sigma should have more pixels than one clipped at 3 sigma"
        )
