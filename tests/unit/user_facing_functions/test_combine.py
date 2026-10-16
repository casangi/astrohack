import os
import pathlib
import shutil

import pytest
import toolviper

from astrohack import combine, open_image
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    execute_cleanup,
    produce_reference_data,
)


class TestCombine:
    data_dir = "combine_data"

    img_name = "ea25_cal_before_reference.image.zarr"

    def_cmb_name = "ea25_cal_before_reference.combine.zarr"
    ref_cmb_name = "ea25_before_reference.combine.zarr"

    ant_id = "ea25"
    ant_key = f"ant_{ant_id}"
    ddi_id = 0
    ddi_key = f"ddi_{ddi_id}"

    cmb_ddi_key = "ddi_99"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""

        toolviper.utils.data.download(file=cls.img_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_cmb_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)

    def test_defaults(self):
        new_cmb_mds = combine(image_name=self.img_name, overwrite=True)
        assert pathlib.Path(self.def_cmb_name).is_dir(), (
            f"A .combine.zarr file named {self.def_cmb_name} does not exist."
        )
        if produce_reference_data():
            return
        ref_cmb_mds = open_image(self.ref_cmb_name)
        assert new_cmb_mds.is_close_to(ref_cmb_mds), (
            "Reference and new mdses are different."
        )

    def test_ddi_and_ant_selection(self):
        """
        Specify a ddi value to be process and check that it is the only one processed.
        """
        if produce_reference_data():
            return
        cmb_mds = combine(
            image_name=self.img_name,
            combine_name=self.def_cmb_name,
            ant="all",
            ddi=self.ddi_id,
            weighted=False,
            parallel=False,
            overwrite=True,
        )

        exp_ddi_list = [self.ddi_key]
        for ant_key, ant_xdt in cmb_mds.items():
            ddi_list = list(ant_xdt.keys())
            assert ddi_list == exp_ddi_list, (
                f"Expected {exp_ddi_list}, but got {ddi_list} for {ant_key}."
            )

        cmb_mds = combine(
            image_name=self.img_name,
            combine_name=self.def_cmb_name,
            ant=self.ant_id,
            ddi="all",
            weighted=False,
            parallel=False,
            overwrite=True,
        )

        exp_ant_list = [self.ant_key]
        assert (list(cmb_mds.keys())) == exp_ant_list, (
            f"Expected {exp_ant_list} but got {list(cmb_mds.keys())}"
        )
        exp_ddi_list = [self.cmb_ddi_key]
        for ant_key, ant_xdt in cmb_mds.items():
            ddi_list = list(ant_xdt.keys())
            assert ddi_list == exp_ddi_list, (
                f"Expected {exp_ddi_list}, but got {ddi_list} for {ant_key}."
            )

    def test_combine_overwrite(self):
        """
        Specify that the output file should be overwritten if it exists; check that it is overwritten.
        """
        if produce_reference_data():
            return
        initial_time = os.path.getctime(self.def_cmb_name)

        # Combine image data
        combine(
            image_name=self.img_name,
            ant=self.ant_id,
            ddi="all",
            weighted=False,
            parallel=False,
            overwrite=True,
        )

        final_time = os.path.getctime(self.def_cmb_name)

        assert initial_time != final_time, (
            "Recreated file has to have a different time from the original file."
        )

        with pytest.raises(FileExistsError):
            combine(
                image_name=self.img_name,
                ant=self.ant_id,
                ddi="all",
                weighted=False,
                parallel=False,
                overwrite=False,
            )
