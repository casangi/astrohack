import os
import pathlib
import shutil

import pytest
import toolviper

from astrohack import open_position
from astrohack.locit import locit
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_lists_equal,
    execute_cleanup,
    produce_reference_data,
)


class TestLocit:
    data_dir = "locit_data"

    lct_name = "locit-input-pha-reference.locit.zarr"

    def_pos_name = "locit-input-pha-reference.position.zarr"
    ref_pos_name = "locit-reference.position.zarr"

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
        toolviper.utils.data.download(cls.lct_name, folder=cls.data_dir)
        toolviper.utils.data.download(cls.ref_pos_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)

    def test_defaults(self):
        """
        Run locit with a specified locit_name and expect a file to be created on disk.
        """

        new_pos_mds = locit(locit_name=self.lct_name, overwrite=True)
        assert pathlib.Path(self.def_pos_name).is_dir(), (
            f"A .position.zarr file named {self.def_pos_name} does not exist."
        )

        if produce_reference_data():
            return

        ref_pos_mds = open_position(self.ref_pos_name)
        assert new_pos_mds.is_close_to(ref_pos_mds), (
            "Reference and new mdses are different."
        )

    def test_data_selection(self):
        """
            Run locit with an antenna id and create a file on disk containing delays and position solutions only \
            from that antenna id.
        """
        if produce_reference_data():
            return
        new_pos_mds = locit(
            locit_name=self.lct_name,
            position_name=self.def_pos_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            combine_ddis="no",
            parallel=False,
            overwrite=True,
        )

        ant_list = list(new_pos_mds.keys())
        assert len(ant_list) == 1, "A single antenna should be present."
        assert ant_list[0] == self.ant_key, (
            "Ant name should be the same as the one given."
        )

        ddi_list = list(new_pos_mds[self.ant_key].keys())
        assert len(ddi_list) == 1, "A single ddi should be present."
        assert ddi_list[0] == self.ddi_key, (
            "DDI key should be the same as the one given."
        )

    def test_fit_kterm(self):
        """
        Run locit with fit_kterm=True and expect a file to be created on disk containing a solution for the kterm.
        """
        if produce_reference_data():
            return
        position_mds = locit(
            locit_name=self.lct_name,
            position_name=self.def_pos_name,
            fit_kterm=True,
            combine_ddis="no",
            parallel=False,
            overwrite=True,
        )

        for ant in position_mds.keys():
            for ddi in position_mds[ant].keys():
                # This is a bit redundant since calling koff_fit when it doesn't exist throws and exception, but it
                # makes what is happening more obvious to others.
                try:
                    position_mds[ant][ddi].koff_fit
                except KeyError as error:
                    raise KeyError(error)

    def test_fit_rate(self):
        """
            Run locit with fit_rate=False and check that the file created on disk contains no solution for the \
            delay rate.
        """
        if produce_reference_data():
            return
        position_mds = locit(
            locit_name=self.lct_name,
            position_name=self.def_pos_name,
            fit_delay_rate=True,
            combine_ddis="no",
            parallel=False,
            overwrite=True,
        )

        for ant in position_mds.keys():
            for ddi in position_mds[ant].keys():
                # This is a bit redundant since calling rate_fit when it doesn't exist throws and exception, but it
                # makes what is happening more obvious to others.
                try:
                    position_mds[ant]["ddi_0"].rate_fit

                except Exception as error:
                    raise Exception(error)

    def test_elevation_limit(self):
        """
        Run locit with elevation_limit=90 and expect locit to fail because there is no available data.
        """
        if produce_reference_data():
            return
        new_pos_mds = locit(
            locit_name=self.lct_name,
            position_name=self.def_pos_name,
            elevation_limit=90.0,
            parallel=False,
            overwrite=True,
        )

        assert new_pos_mds is None, (
            "There should be no position mds created when elevation limit is 90 degrees"
        )

    def test_polarization(self):
        """
        Run locit with polarization='R' and check that the file created on disk contains only delays for R.
        """
        if produce_reference_data():
            return
        pol_sel = "R"
        position_mds = locit(
            locit_name=self.lct_name,
            position_name=self.def_pos_name,
            polarization=pol_sel,
            parallel=False,
            overwrite=True,
        )

        for ant in position_mds.keys():
            assert position_mds[ant].polarization == pol_sel

    def test_combine_ddis(self):
        """
          Run locit with combine_ddis=False and check that the file created on disk contains delays and position \
          solutions for all DDIs.
        """
        if produce_reference_data():
            return
        position_mds = locit(
            locit_name=self.lct_name,
            position_name=self.def_pos_name,
            combine_ddis="simple",
            parallel=False,
            overwrite=True,
        )

        ref_list = [
            "DECLINATION",
            "DELAYS",
            "ELEVATION",
            "HOUR_ANGLE",
            "LST",
            "MODEL",
            "SCANS",
        ]
        for key in position_mds.keys():
            assert are_lists_equal(list(position_mds[key].keys()), ref_list), (
                "An XDS should contain all the expected keys"
            )

    def test_scan_exclusion(self):
        """
        Test excluding a few scans from the data set.
        """
        if produce_reference_data():
            return
        bad_scans = [3, 5, 15]
        position_mds = locit(
            locit_name=self.lct_name,
            position_name=self.def_pos_name,
            combine_ddis="simple",
            exclude_scans=bad_scans,
            parallel=False,
            overwrite=True,
        )
        ant_xds = position_mds[self.ant_key]
        for bad_scan in bad_scans:
            assert bad_scan not in ant_xds.SCANS.values, (
                f"Scan {bad_scan} should have been excluded from dataset"
            )

    def test_overwrite(self):
        """
        Specify the output file should be overwritten; check that it WAS.
        """
        if produce_reference_data():
            return
        # To check this properly we need to not only know an exception was not thrown but that the file is ACTUALLY
        # overwritten. We do this by checking the modification time.
        initial_time = os.path.getctime(self.def_pos_name)

        locit(
            locit_name=self.lct_name,
            overwrite=True,
        )
        modified_time = os.path.getctime(self.def_pos_name)
        assert initial_time != modified_time, (
            "Recreated file has to have a different time from the original file."
        )

        with pytest.raises(FileExistsError):
            locit(
                locit_name=self.lct_name,
                overwrite=False,
            )
