import shutil

import matplotlib
from toolviper.utils import data

from astrohack import AstrohackHologFile, open_holog
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_png_files_close,
    are_txt_files_equal,
    execute_cleanup,
    produce_reference_data,
)

matplotlib.use("Agg")


class TestHologMDS:
    data_dir = "holog_data"
    destination_folder = "holog_exports"
    ref_products_name = "ref_holog_products"

    hlg_name = "ea25_cal_small_before_reference.holog.zarr"

    ant_id = "ea25"
    ant_key = f"ant_{ant_id}"
    ddi_id = 0
    ddi_key = f"ddi_{ddi_id}"
    map_id = 0
    map_key = f"map_{map_id}"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.hlg_name, folder=cls.data_dir)
        data.download(file=cls.ref_products_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)
            shutil.rmtree(cls.destination_folder)
        return

    def test_init(self):
        hlg_mds = AstrohackHologFile(self.hlg_name)
        assert isinstance(hlg_mds, AstrohackHologFile)

    def test_text_exports(self):
        hlg_mds = open_holog(self.hlg_name)

        uvhol_like_file_name = "holog_visibilities_ant_ea25_ddi_0_map_0.txt"
        hlg_mds.export_to_aips(
            self.destination_folder,
            ant=self.ant_id,
            ddi=self.ddi_id,
            map_id=self.map_id,
            parallel=False,
        )
        obs_summ_file_name = "obs_summ.txt"
        hlg_mds.observation_summary(
            f"{self.destination_folder}/{obs_summ_file_name}",
            ant=self.ant_id,
            ddi=self.ddi_id,
            map_id=self.map_id,
            print_summary=False,
            parallel=False,
        )
        if produce_reference_data():
            return

        assert are_txt_files_equal(
            f"{self.ref_products_name}/{uvhol_like_file_name}",
            f"{self.destination_folder}/{uvhol_like_file_name}",
            ignored_key_words=["DATE-OBS"],
        ), "AIPS like export is different from reference"
        assert are_txt_files_equal(
            f"{self.destination_folder}/{obs_summ_file_name}",
            f"{self.ref_products_name}/{obs_summ_file_name}",
        ), "Observation summary is different from reference"

    def test_plot_exports(self):
        hlg_mds = open_holog(self.hlg_name)

        hlg_mds.plot_diagnostics(
            self.destination_folder,
            ant=self.ant_id,
            ddi=self.ddi_id,
            map_id=self.map_id,
            parallel=False,
        )
        hlg_mds.plot_lm_sky_coverage(
            self.destination_folder,
            ant=self.ant_id,
            ddi=self.ddi_id,
            map_id=self.map_id,
            plot_correlation="RR",
            parallel=False,
        )
        if produce_reference_data():
            return

        plot_name = "holog_diagnostics_ant_ea25_ddi_0_map_0.png"
        assert are_png_files_close(
            f"{self.destination_folder}/{plot_name}",
            f"{self.ref_products_name}/{plot_name}",
        ), "Calibration diagnostics plot is different from reference"

        plot_name = "holog_directional_cosines_ant_ea25_ddi_0_map_0.png"
        assert are_png_files_close(
            f"{self.destination_folder}/{plot_name}",
            f"{self.ref_products_name}/{plot_name}",
        ), "Directional_cosines plot is different from reference"

        plot_name = "holog_directional_cosines_RR_ant_ea25_ddi_0_map_0.png"
        assert are_png_files_close(
            f"{self.destination_folder}/{plot_name}",
            f"{self.ref_products_name}/{plot_name}",
        ), "Correlation vs Directional cosines plot is different from reference"
