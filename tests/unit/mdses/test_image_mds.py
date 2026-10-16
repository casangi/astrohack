import glob
import os
import pathlib
import shutil

import matplotlib
from toolviper.utils import data

from astrohack import AstrohackImageFile, open_image
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_fits_files_close,
    are_png_files_close,
    are_txt_files_equal,
    execute_cleanup,
    produce_reference_data,
)

matplotlib.use("Agg")


class TestimageMDS:
    data_dir = "image_data"
    destination_folder = "image_exports"
    ref_products_name = "ref_image_products"

    img_name = "ea25_cal_before_reference.image.zarr"

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
        data.download(file=cls.img_name, folder=cls.data_dir)
        data.download(file=cls.ref_products_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)
            shutil.rmtree(cls.destination_folder)
        return

    def test_init(self):
        img_mds = AstrohackImageFile(self.img_name)
        assert isinstance(img_mds, AstrohackImageFile)

    def test_fits_exports(self):
        img_mds = open_image(self.img_name)
        img_mds.export_to_fits(
            self.destination_folder, ant=self.ant_id, ddi=self.ddi_id, parallel=False
        )
        if produce_reference_data():
            return
        self.fits_list_assertions()
        return

    def test_text_exports(self):
        img_mds = open_image(self.img_name)

        exec_dict = {
            f"image_phase_fit_ant_{self.ant_id}_ddi_{self.ddi_id}.txt": img_mds.export_phase_fit_results,
            f"image_zernike_fit_ant_{self.ant_id}_ddi_{self.ddi_id}.txt": img_mds.export_zernike_fit_results,
        }
        for txt_file_name, func in exec_dict.items():
            func(
                self.destination_folder,
                ant=self.ant_id,
                ddi=self.ddi_id,
                parallel=False,
            )
            if not produce_reference_data():
                assert are_txt_files_equal(
                    f"{self.destination_folder}/{txt_file_name}",
                    f"{self.ref_products_name}/{txt_file_name}",
                ), "Phase fit results are different from reference"

        obs_summ_file_name = "obs_summ.txt"
        img_mds.observation_summary(
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
        img_mds = open_image(self.img_name)
        methods = [
            img_mds.plot_apertures,
            img_mds.plot_zernike_model,
            img_mds.plot_beams,
        ]

        for method in methods:
            method(
                self.destination_folder,
                ant=self.ant_id,
                ddi=self.ddi_id,
                parallel=False,
            )
            if not produce_reference_data():
                self.plot_list_assertions("image_aperture_*.png")
