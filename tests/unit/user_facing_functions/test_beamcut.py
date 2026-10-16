import glob
import pathlib
import shutil

import matplotlib
import numpy as np
from toolviper.utils import data

from astrohack import beamcut, extract_holog, extract_pointing, open_beamcut
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_lists_equal,
    execute_cleanup,
    produce_reference_data,
)

matplotlib.use("Agg")


def retrieve_data_from_report(report):
    az_val = None
    el_val = None
    azel_unit = None
    lm_unit = None

    with open(report) as rep_file:
        for line in rep_file:
            if line[0] == "-":  # header line
                wrds = line.split()
                az_val = float(wrds[11].split("=")[-1][:-1])
                el_val = float(wrds[12].split("=")[-1])
                azel_unit = wrds[13]
            elif "|" in line:
                wrds = line.split("|")
                center_header = wrds[2]
                lm_unit = center_header.split()[1][1:-1]
                break
    return az_val, el_val, azel_unit, lm_unit


class TestBeamcut:
    data_dir = "beamcut_data"
    destination_folder = "beamcut_exports"

    ms_name = "kband_beamcut_small.ms"
    point_name = "kband_beamcut_small.point.zarr"
    holog_name = "kband_beamcut_small.holog.zarr"
    local_beamcut_name = "kband_beamcut_small_local.beamcut.zarr"
    remote_beamcut_name = "kband_beamcut_small.beamcut.zarr"
    ea15_report = "beamcut_exports/beamcut_report_ant_ea15_ddi_0.txt"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.ms_name, folder=cls.data_dir)
        data.download(file=cls.remote_beamcut_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

        extract_pointing(
            ms_name=cls.ms_name,
            point_name=cls.point_name,
            overwrite=True,
            parallel=False,
        )

        # Extract holography data using holog_obd_dict
        extract_holog(
            ms_name=cls.ms_name,
            point_name=cls.point_name,
            holog_name=cls.holog_name,
            data_column="DATA",  # Beamcut ms is a split, hence we need to use "DATA"
            parallel=False,
            overwrite=True,
        )

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir, ignore_errors=True)
            shutil.rmtree(cls.destination_folder, ignore_errors=True)

    def test_results(self):
        # Has to be run first
        new_bmc_mds = beamcut(
            holog_name=self.holog_name,
            beamcut_name=self.local_beamcut_name,
            ddi=0,
            ant="ea15",
            overwrite=True,
        )
        if produce_reference_data():
            return

        ref_bmc_mds = open_beamcut(self.remote_beamcut_name)
        assertion = ref_bmc_mds.is_close_to(new_bmc_mds)
        assert assertion, "Reference and new mdses are not close enough."

    def test_destination(self):
        # Deleting destination if it exists just to make test more robust
        shutil.rmtree(self.destination_folder, ignore_errors=True)
        if produce_reference_data():
            return

        beamcut(
            holog_name=self.holog_name,
            beamcut_name=self.local_beamcut_name,
            destination=self.destination_folder,
            overwrite=True,
        )

        destination_exists = pathlib.Path(self.destination_folder).is_dir()
        assert destination_exists, (
            f"destination folder {self.destination_folder} does not exist"
        )

        exp_n_pngs = 16
        n_pngs = len(glob.glob(f"{self.destination_folder}/*.png"))
        assert n_pngs == exp_n_pngs, (
            f"When a destination is given beamcut should prepare {exp_n_pngs} pngs, {n_pngs} have been found"
        )

        exp_n_txt = 4
        n_txts = len(glob.glob(f"{self.destination_folder}/*.txt"))
        assert n_txts == exp_n_txt, (
            f"When a destination is given beamcut should prepare {exp_n_txt} reports, {n_txts} have been found"
        )

    def test_data_selection(self):
        # This test depends on knowing the contents of the original ms
        if produce_reference_data():
            return
        beamcut_mds = beamcut(
            holog_name=self.holog_name,
            beamcut_name=self.local_beamcut_name,
            overwrite=True,
            ant="all",
            ddi="all",
        )

        full_ant_list = ["ant_ea15", "ant_ea17"]
        full_ddi_list = ["ddi_0", "ddi_1"]

        mds_ant_list = list(beamcut_mds.keys())
        assert are_lists_equal(full_ant_list, mds_ant_list), (
            'With ant="all", mds_ant_list should be equal to full_ant_list'
        )

        for ant in full_ant_list:
            ddi_list = list(beamcut_mds[ant].keys())
            assert are_lists_equal(ddi_list, full_ddi_list), (
                'With ddi="all", ddi_list should be equal to full_ddi_list'
            )

        beamcut_mds = beamcut(
            holog_name=self.holog_name,
            beamcut_name=self.local_beamcut_name,
            overwrite=True,
            ant="ea15",
            ddi=1,
        )

        short_ant_list = ["ant_ea15"]
        short_ddi_list = ["ddi_1"]

        mds_ant_list = list(beamcut_mds.keys())
        assert are_lists_equal(short_ant_list, mds_ant_list), (
            'With ant="all", mds_ant_list should be equal to short_ant_list'
        )

        for ant in short_ant_list:
            ddi_list = list(beamcut_mds[ant].keys())
            assert are_lists_equal(ddi_list, short_ddi_list), (
                'With ddi="all", ddi_list should be equal to short_ddi_list'
            )

    def test_report_configuration(self):
        if produce_reference_data():
            return
        # this test depends on us knowing some values expected to be in the report
        beamcut(
            holog_name=self.holog_name,
            beamcut_name=self.local_beamcut_name,
            destination=self.destination_folder,
            ant="ea15",
            ddi=0,
            overwrite=True,
        )

        rep_az, rep_el, rep_azel_unit, rep_lm_unit = retrieve_data_from_report(
            self.ea15_report
        )
        exp_az = 294.3
        exp_el = 45.5
        exp_azel_unit = "deg"
        exp_lm_unit = "amin"
        assert np.isclose(rep_az, exp_az, atol=1e-1), (
            f"Report's azimuth should be {exp_az} {exp_azel_unit}, got {rep_az} {rep_azel_unit}"
        )
        assert np.isclose(rep_el, exp_el, atol=1e-1), (
            f"Report's elevation should be {exp_el} {exp_azel_unit}, got {rep_el} {rep_azel_unit}"
        )
        assert rep_azel_unit == exp_azel_unit, (
            f"Report's azimuth/elevation unit should be {exp_azel_unit}, got {rep_azel_unit}"
        )
        assert rep_lm_unit == exp_lm_unit, (
            f"Report's lm offsets unit should be {exp_lm_unit}, got {rep_lm_unit}"
        )

        # this test depends on us knowing some values expected to be in the report
        beamcut(
            holog_name=self.holog_name,
            beamcut_name=self.local_beamcut_name,
            destination=self.destination_folder,
            ant="ea15",
            ddi=0,
            azel_unit="amin",
            lm_unit="asec",
            overwrite=True,
        )

        rep_az, rep_el, rep_azel_unit, rep_lm_unit = retrieve_data_from_report(
            self.ea15_report
        )

        exp_az = 294.0 * 60
        exp_el = 46.0 * 60
        exp_azel_unit = "amin"
        exp_lm_unit = "asec"
        assert np.isclose(rep_az, exp_az, atol=30), (
            f"Report's azimuth should be {exp_az} {exp_azel_unit}, got {rep_az} {rep_azel_unit}"
        )
        assert np.isclose(rep_el, exp_el, atol=30), (
            f"Report's elevation should be {exp_el} {exp_azel_unit}, got {rep_el} {rep_azel_unit}"
        )
        assert rep_azel_unit == exp_azel_unit, (
            f"Report's azimuth/elevation unit should be {exp_azel_unit}, got {rep_azel_unit}"
        )
        assert rep_lm_unit == exp_lm_unit, (
            f"Report's lm offsets unit should be {exp_lm_unit}, got {rep_lm_unit}"
        )

    def test_naming(self):
        shutil.rmtree(self.remote_beamcut_name, ignore_errors=True)
        if produce_reference_data():
            return

        # has to be run last!
        beamcut(
            holog_name=self.holog_name,
            overwrite=True,
        )

        assert pathlib.Path(self.remote_beamcut_name).is_dir(), (
            "If no beamcut_name is given, beamcut should create an output file named {self.remote_beamcut_name}"
        )

        crazy_name = self.data_dir + "/crazy_name.beamcut.zarr"

        beamcut(
            holog_name=self.holog_name,
            beamcut_name=crazy_name,
            overwrite=True,
        )

        assert pathlib.Path(crazy_name).is_dir(), (
            f"Beamcut should create an output file named {crazy_name}"
        )

        return
