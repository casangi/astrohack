import glob
import os
import pathlib
import shutil

import matplotlib
from toolviper.utils import data

from astrohack import AstrohackPanelFile, open_panel
from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_fits_files_close,
    are_png_files_close,
    are_txt_files_equal,
    execute_cleanup,
    produce_reference_data,
)

matplotlib.use("Agg")


class TestpanelMDS:
    data_dir = "panel_data"
    destination_folder = "panel_exports"
    ref_products_name = "ref_panel_products"

    pnl_name = "ea25_before_reference.panel.zarr"

    ant_id = "ea25"
    ant_key = f"ant_{ant_id}"
    ddi_id = 0
    ddi_key = f"ddi_{ddi_id}"

    def plot_list_assertions(self, pattern):
        plot_name_list = [
            os.path.basename(full_path_name)
            for full_path_name in glob.glob(f"{self.ref_products_name}/{pattern}")
        ]
        for plot_name in plot_name_list:
            assert pathlib.Path(f"{self.destination_folder}/{plot_name}").is_file(), (
                f"{plot_name} was not created"
            )
            assert are_png_files_close(
                f"{self.destination_folder}/{plot_name}",
                f"{self.ref_products_name}/{plot_name}",
            ), f"{plot_name} Differs from reference"

    def fits_list_assertions(self):
        fits_list = [
            os.path.basename(full_path_name)
            for full_path_name in glob.glob(f"{self.ref_products_name}/*.fits")
        ]
        for fits_name in fits_list:
            assert pathlib.Path(f"{self.destination_folder}/{fits_name}").is_file(), (
                f"{fits_name} was not created"
            )
            assert are_fits_files_close(
                f"{self.destination_folder}/{fits_name}",
                f"{self.ref_products_name}/{fits_name}",
            ), f"{fits_name} is different from reference"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.pnl_name, folder=cls.data_dir)
        data.download(file=cls.ref_products_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)
            shutil.rmtree(cls.destination_folder)
        return

    def test_init(self):
        pnl_mds = AstrohackPanelFile(self.pnl_name)
        assert isinstance(pnl_mds, AstrohackPanelFile)

    def test_get_antenna(self):
        pnl_mds = open_panel(self.pnl_name)

        ant_obj = pnl_mds.get_antenna(ant=self.ant_id, ddi=self.ddi_id)
        assert isinstance(ant_obj, AntennaSurface), (
            "Gotten antenna object is not a Antenna Surface obj"
        )

    def test_fits_exports(self):
        pnl_mds = open_panel(self.pnl_name)
        pnl_mds.export_to_fits(
            self.destination_folder, ant=self.ant_id, ddi=self.ddi_id, parallel=False
        )
        if not produce_reference_data():
            self.fits_list_assertions()
        return

    def test_text_exports(self):
        pnl_mds = open_panel(self.pnl_name)
        pnl_mds.export_gain_tables(
            self.destination_folder, ant=self.ant_id, ddi=self.ddi_id, parallel=False
        )
        txt_file_name = "panel_gains_ant_ea25_ddi_0.txt"
        if not produce_reference_data():
            assert are_txt_files_equal(
                f"{self.destination_folder}/{txt_file_name}",
                f"{self.ref_products_name}/{txt_file_name}",
            ), "Gain tables are different from reference"

        obs_summ_file_name = "obs_summ.txt"
        pnl_mds.observation_summary(
            f"{self.destination_folder}/{obs_summ_file_name}",
            ant=self.ant_id,
            ddi=self.ddi_id,
            parallel=False,
            print_summary=False,
        )
        if not produce_reference_data():
            assert are_txt_files_equal(
                f"{self.destination_folder}/{obs_summ_file_name}",
                f"{self.ref_products_name}/{obs_summ_file_name}",
            ), "Observation summary is different from reference"

        return

    def test_plot_exports(self):
        pnl_mds = open_panel(self.pnl_name)

        pnl_mds.plot_antennas(
            self.destination_folder,
            ant=self.ant_id,
            ddi=self.ddi_id,
            parallel=False,
            plot_type="all",
        )

        pnl_mds.export_screws(self.destination_folder, ant=self.ant_id, ddi=self.ddi_id)

        if not produce_reference_data():
            self.plot_list_assertions("panel_*.png")
            screws_file_name = "panel_screws_ant_ea25_ddi_0.txt"
            assert are_txt_files_equal(
                f"{self.destination_folder}/{screws_file_name}",
                f"{self.ref_products_name}/{screws_file_name}",
            ), "Observation summary is different from reference"
        return
