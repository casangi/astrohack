import pathlib
import shutil

import toolviper
from toolviper.dask.client import local_client

from astrohack import (
    beamcut,
    extract_holog,
    extract_pointing,
    holog,
    locit,
    open_holog,
    open_image,
    open_panel,
    open_pointing,
    open_position,
    panel,
)
from astrohack.utils.verification_tools import add_data_folder_to_names_in_class


class TestAstrohackInParallel:
    data_dir = "parallel_data"
    ms_name = "ea25_cal_small_before_fixed.split.ms"

    def_pnt_name = "ea25_cal_small_before_fixed.split.point.zarr"
    ref_pnt_name = "ea25_cal_small_before_reference.point.zarr"

    def_hlg_name = "ea25_cal_small_before_fixed.split.holog.zarr"
    ref_hlg_name = "ea25_cal_small_before_reference.holog.zarr"

    def_img_name = "ea25_cal_small_before_reference.image.zarr"
    ref_img_name = "ea25_cal_before_reference.image.zarr"

    def_pnl_name = "ea25_cal_before_reference.panel.zarr"
    ref_pnl_name = "ea25_before_reference.panel.zarr"

    lct_name = "locit-input-pha-reference.locit.zarr"

    def_pos_name = "locit-input-pha-reference.position.zarr"
    ref_pos_name = "locit-reference.position.zarr"

    bmc_ms_name = "kband_beamcut_small.ms"

    bmc_pnt_name = "kband_beamcut_small.point.zarr"
    bmc_hlg_name = "kband_beamcut_small.holog.zarr"

    def_bmc_name = "kband_beamcut_small_local.beamcut.zarr"
    ref_bmc_name = "kband_beamcut_small.beamcut.zarr"

    client = None

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        toolviper.utils.data.download(file=cls.ms_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_pnt_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_hlg_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_img_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_pnl_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.lct_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_pos_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.bmc_ms_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

        cls.client = local_client(cores=4, memory_limit="2GB")

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_dir)
        cls.client.close()

    def test_extract_pointing(self):
        new_pnt_mds = extract_pointing(
            ms_name=self.ms_name,
            point_name=self.def_pnt_name,
            overwrite=True,
            parallel=True,
        )
        assert pathlib.Path(self.def_pnt_name).is_dir(), (
            f"A .point.zarr file named {self.def_pnt_name} does not exist."
        )

        ref_pnt_mds = open_pointing(self.ref_pnt_name)
        assert new_pnt_mds.is_close_to(ref_pnt_mds), (
            "Reference and new mdses are different."
        )

    def test_extract_holog(self):
        new_hlg_mds = extract_holog(
            ms_name=self.ms_name,
            point_name=self.ref_pnt_name,
            holog_name=self.def_hlg_name,
            overwrite=True,
            parallel=True,
        )
        assert pathlib.Path(self.def_hlg_name).is_dir(), (
            f"A .holog.zarr file named {self.def_hlg_name} does not exist."
        )

        ref_hlg_mds = open_holog(self.ref_hlg_name)
        assert new_hlg_mds.is_close_to(ref_hlg_mds), (
            "Reference and new mdses are different."
        )

    def test_holog(self):
        new_img_mds = holog(
            holog_name=self.ref_hlg_name,
            image_name=self.def_img_name,
            overwrite=True,
            parallel=True,
        )
        assert pathlib.Path(self.def_img_name).is_dir(), (
            f"A .image.zarr file named {self.def_img_name} does not exist."
        )

        ref_img_mds = open_image(self.ref_img_name)
        assert new_img_mds.is_close_to(ref_img_mds), (
            "Reference and new mdses are different."
        )

    def test_panel(self):
        new_pnl_mds = panel(
            image_name=self.ref_img_name,
            panel_name=self.def_pnl_name,
            overwrite=True,
            parallel=True,
        )
        assert pathlib.Path(self.def_pnl_name).is_dir(), (
            f"A .panel.zarr file named {self.def_pnl_name} does not exist."
        )

        ref_pnl_mds = open_panel(self.ref_pnl_name)
        assert new_pnl_mds.is_close_to(ref_pnl_mds), (
            "Reference and new mdses are different."
        )

    def test_locit(self):
        new_pos_mds = locit(
            locit_name=self.lct_name,
            position_name=self.def_pos_name,
            parallel=True,
            overwrite=True,
        )
        assert pathlib.Path(self.def_pos_name).is_dir(), (
            f"A .position.zarr file named {self.def_pos_name} does not exist."
        )

        ref_pos_mds = open_position(self.ref_pos_name)
        assert new_pos_mds.is_close_to(ref_pos_mds), (
            "Reference and new mdses are different."
        )

    def test_beamcut(self):
        extract_pointing(
            ms_name=self.bmc_ms_name,
            point_name=self.bmc_pnt_name,
            overwrite=True,
            parallel=True,
        )

        extract_holog(
            ms_name=self.bmc_ms_name,
            point_name=self.bmc_pnt_name,
            holog_name=self.bmc_hlg_name,
            data_column="DATA",
            overwrite=True,
            parallel=True,
        )

        parallel_mds = beamcut(
            holog_name=self.bmc_hlg_name,
            beamcut_name=self.def_bmc_name,
            parallel=True,
            overwrite=True,
            destination=None,
        )
        assert pathlib.Path(self.def_bmc_name).is_dir(), (
            f"A .beamcut.zarr file named {self.def_bmc_name} does not exist."
        )

        serial_mds = beamcut(
            holog_name=self.bmc_hlg_name,
            beamcut_name=f"{self.data_dir}/serial.beamcut.zarr",
            parallel=False,
            overwrite=True,
            destination=None,
        )

        assert parallel_mds.is_close_to(serial_mds), (
            "parallel and serial mdses are different."
        )
