import os
import shutil

import matplotlib
import pytest
from toolviper.utils import data

from astrohack import AstrohackLocitFile, open_locit
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_lists_equal,
    are_png_files_close,
    capture_prints_from_function,
    execute_cleanup,
    is_captured_output_equal_to_txt_reference,
    produce_reference_data,
)

matplotlib.use("Agg")


class TestLocitMDS:
    data_dir = "locit_data"
    destination_folder = "locit_exports"
    ref_products_name = "ref_locit_products"

    locit_name = "locit-input-pha-reference.locit.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.locit_name, folder=cls.data_dir)
        data.download(file=cls.ref_products_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir, ignore_errors=True)
            shutil.rmtree(cls.destination_folder, ignore_errors=True)
        return

    def test_locit_mds_init(self):
        locit_mds = AstrohackLocitFile(self.locit_name)
        assert isinstance(locit_mds, AstrohackLocitFile)

    def test_locit_mds_text_exports(self):
        locit_mds = open_locit(self.locit_name)

        exec_dict = {
            "src_tab_reference.txt": locit_mds.print_source_table,
            "array_cfg_reference.txt": locit_mds.print_array_configuration,
        }
        for txt_file_name, func in exec_dict.items():
            if produce_reference_data():
                os.makedirs(f"{self.destination_folder}", exist_ok=True)
                output = capture_prints_from_function(func)
                with open(
                    f"{self.destination_folder}/{txt_file_name}", "w"
                ) as out_file:
                    out_file.write(output)
            else:
                assert is_captured_output_equal_to_txt_reference(
                    func, f"{self.ref_products_name}/{txt_file_name}"
                ), f"{txt_file_name} should be exactly equal to reference"

    @pytest.mark.skip(
        reason="Plot comparison is flaky and cannot be trusted to yield consistent results"
    )
    def test_locit_mds_plot_exports(self):
        locit_mds = open_locit(self.locit_name)

        src_fk5_plot_name = "locit_source_table_fk5.png"
        locit_mds.plot_source_positions(self.destination_folder, precessed=False)
        if not produce_reference_data():
            equal, msg = are_png_files_close(
                f"{self.destination_folder}/{src_fk5_plot_name}",
                f"{self.ref_products_name}/{src_fk5_plot_name}",
            )
            assert equal, (
                f"{msg}: FK5 source position plot should be exactly equal to reference FK5 source position plot"
            )

        src_prece_plot_name = "locit_source_table_precessed.png"
        locit_mds.plot_source_positions(self.destination_folder, precessed=True)
        if not produce_reference_data():
            equal, msg = are_png_files_close(
                f"{self.destination_folder}/{src_prece_plot_name}",
                f"{self.ref_products_name}/{src_prece_plot_name}",
            )
            assert equal, (
                f"{msg}: Precessed source position plot should be exactly equal to reference precessed source position plot"
            )

        array_cfg_plot_name = "locit_array_configuration.png"
        locit_mds.plot_array_configuration(self.destination_folder)
        if not produce_reference_data():
            equal, msg = are_png_files_close(
                f"{self.destination_folder}/{array_cfg_plot_name}",
                f"{self.ref_products_name}/{array_cfg_plot_name}",
            )
            assert equal, (
                f"{msg}: Array configuration plot should be exactly equal to reference array configuration plot"
            )

    def test_locit_mds_metadata_style(self):
        locit_mds = open_locit(self.locit_name)

        assert "source_dict" in list(locit_mds.root.attrs.keys()), (
            "Root attributes should contain 'source_dict'"
        )

        expected_src_keys = ["fk5", "id", "name", "precessed"]
        src_table = locit_mds.root.attrs["source_dict"]
        for key, value in src_table.items():
            assert key.isdigit(), "Source key should be a digit referencing field Ids"
            assert are_lists_equal(list(value.keys()), expected_src_keys), (
                "Source position keys should be the same as expected keys"
            )

        expected_ant_keys = [
            "geocentric_position",
            "id",
            "latitude",
            "longitude",
            "name",
            "offset",
            "radius",
            "reference",
            "station",
        ]
        for ant_xdtree in locit_mds.values():
            assert "antenna_info" in list(ant_xdtree.attrs.keys()), (
                "Each antenna xarray DataTree needs to contain antenna info"
            )
            antenna_info = ant_xdtree.attrs["antenna_info"]
            assert are_lists_equal(list(antenna_info.keys()), expected_ant_keys)
