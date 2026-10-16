import os
import pathlib
import shutil

import pytest
import toolviper

from astrohack import open_position
from astrohack.fringefit_locit import fringefit_locit
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    execute_cleanup,
    produce_reference_data,
)


class TestFringeFitLocit:
    data_dir = "fringefit_locit_data"

    fringefit_name = "fft_locit_input.sbd"

    def_pos_name = "fft_locit_input.position.zarr"
    alt_pos_name = "fft_locit_alt.position.zarr"
    ref_pos_name = "fft_locit_reference.position.zarr"

    ant_id = "ea21"
    ant_key = f"ant_{ant_id}"
    ddi_id = 1

    @classmethod
    def setup_class(cls):
        """
        Setup any state specific to the execution of the given test class
        such as fetching test data
        """
        toolviper.utils.data.download(cls.fringefit_name, folder=cls.data_dir)
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

        new_pos_mds = fringefit_locit(
            fringefit_caltable=self.fringefit_name, overwrite=True
        )
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
        new_pos_mds = fringefit_locit(
            fringefit_caltable=self.fringefit_name,
            position_name=self.alt_pos_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            parallel=False,
            overwrite=True,
        )

        ant_list = list(new_pos_mds.keys())
        assert len(ant_list) == 1, "A single antenna should be present."
        assert ant_list[0] == self.ant_key, (
            "Ant name should be the same as the one given."
        )

        ref_pos_mds = open_position(self.ref_pos_name)

        ref_ave_freq = ref_pos_mds[self.ant_key].attrs["frequency"]
        new_ave_freq = new_pos_mds[self.ant_key].attrs["frequency"]
        assert ref_ave_freq != new_ave_freq, (
            "Average frequencies in new and ref position mds must not match"
        )

        ref_delay_arr = ref_pos_mds[self.ant_key]["DELAYS"]
        new_delay_arr = new_pos_mds[self.ant_key]["DELAYS"]
        assert new_delay_arr.size < ref_delay_arr.size, (
            "New delay array must be smaller than ref delay array."
        )

    def test_fit_kterm(self):
        """
        Run locit with fit_kterm=True and expect a file to be created on disk containing a solution for the kterm.
        """
        if produce_reference_data():
            return
        position_mds = fringefit_locit(
            fringefit_caltable=self.fringefit_name,
            position_name=self.def_pos_name,
            fit_kterm=True,
            parallel=False,
            overwrite=True,
        )

        for ant in position_mds.keys():
            try:
                position_mds[ant].koff_fit
            except KeyError as error:
                raise KeyError(error)

    def test_fit_rate(self):
        """
            Run locit with fit_rate=False and check that the file created on disk contains no solution for the \
            delay rate.
        """
        if produce_reference_data():
            return
        position_mds = fringefit_locit(
            fringefit_caltable=self.fringefit_name,
            position_name=self.def_pos_name,
            fit_delay_rate=True,
            parallel=False,
            overwrite=True,
        )

        for ant in position_mds.keys():
            try:
                position_mds[ant].rate_fit

            except Exception as error:
                raise Exception(error)

    def test_elevation_limit(self):
        """
        Run locit with elevation_limit=90 and expect locit to fail because there is no available data.
        """
        if produce_reference_data():
            return
        new_pos_mds = fringefit_locit(
            fringefit_caltable=self.fringefit_name,
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
        for pol in ["R", "L"]:
            position_mds = fringefit_locit(
                fringefit_caltable=self.fringefit_name,
                position_name=self.def_pos_name,
                polarization=pol,
                parallel=False,
                overwrite=True,
            )

            for ant in position_mds.keys():
                assert position_mds[ant].polarization == pol

    def test_scan_exclusion(self):
        """
        Test excluding a few scans from the data set.
        """
        if produce_reference_data():
            return
        bad_scans = [3, 5, 15]
        position_mds = fringefit_locit(
            fringefit_caltable=self.fringefit_name,
            position_name=self.def_pos_name,
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

        fringefit_locit(
            fringefit_caltable=self.fringefit_name,
            overwrite=True,
        )
        modified_time = os.path.getctime(self.def_pos_name)
        assert initial_time != modified_time, (
            "Recreated file has to have a different time from the original file."
        )

        with pytest.raises(FileExistsError):
            fringefit_locit(
                fringefit_caltable=self.fringefit_name,
                overwrite=False,
            )
