import os
import pathlib
import shutil

import pytest
import toolviper

from astrohack import open_locit
from astrohack.extract_locit import extract_locit
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    execute_cleanup,
    produce_reference_data,
)


class TestExtractLocit:
    data_dir = "extract_locit_data"
    cal_table_name = "locit-input-pha.cal"

    def_lct_name = "locit-input-pha.locit.zarr"
    ref_lct_name = "locit-input-pha-reference.locit.zarr"

    ant_id = "ea17"
    ant_key = f"ant_{ant_id}"
    ddi_id = 0
    ddi_key = f"ddi_{ddi_id}"

    @classmethod
    def setup_class(cls):
        """
        Setup any state specific to the execution of the given test class
        such as fetching test data
        """

        toolviper.utils.data.download(file=cls.cal_table_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_lct_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)

    def test_defaults(self):
        """
        Create locit file with a cal-table and check it is created correctly.
        """

        # Create locit_mds and check the dictionary structure
        new_lct_mds = extract_locit(cal_table=self.cal_table_name, overwrite=True)
        assert pathlib.Path(self.def_lct_name).is_dir(), (
            f"A .locit.zarr file named {self.def_lct_name} does not exist."
        )

        if produce_reference_data():
            return

        ref_lct_mds = open_locit(self.ref_lct_name)
        assert new_lct_mds.is_close_to(ref_lct_mds), (
            "Reference and new mdses are different."
        )

    def test_data_selection(self):
        """
        Check that only specified antenna is processed.
        """
        if produce_reference_data():
            return

        new_lct_mds = extract_locit(
            cal_table=self.cal_table_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
        )

        ant_list = list(new_lct_mds.keys())
        assert len(ant_list) == 1, "A single antenna should be present."
        assert ant_list[0] == self.ant_key, (
            "Ant name should be the same as the one given."
        )

        ddi_list = list(new_lct_mds[self.ant_key].keys())
        assert len(ddi_list) == 1, "A single ddi should be present."
        assert ddi_list[0] == self.ddi_key, (
            "DDI key should be the same as the one given."
        )

    def test_overwrite(self):
        """
        Specify the output file should be overwritten; check that it WAS.
        """
        if produce_reference_data():
            return

        # To check this properly we need to not only know an exception was not thrown but that the file is ACTUALLY
        # overwritten. We do this by checking the modification time.
        initial_time = os.path.getctime(self.def_lct_name)

        extract_locit(
            cal_table=self.cal_table_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
        )
        modified_time = os.path.getctime(self.def_lct_name)
        assert initial_time != modified_time, (
            "Recreated file has to have a different time from the original file."
        )

        with pytest.raises(FileExistsError):
            extract_locit(
                cal_table=self.cal_table_name,
                ant=self.ant_id,
                ddi=self.ddi_id,
                overwrite=False,
            )
