import shutil

import matplotlib
import pytest
from toolviper.utils import data

from astrohack import (
    AstrohackLocitFile,
    AstrohackPositionFile,
    extract_locit,
    locit,
    open_position,
)
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_png_files_close,
    are_txt_files_equal,
    execute_cleanup,
    produce_reference_data,
)

matplotlib.use("Agg")


class TestPositionMDS:
    data_dir = "position_data"
    destination_folder = "position_exports"
    ref_products_name = "ref_position_products"

    phase_cal_table_name = "locit-input-pha.cal"
    locit_name = "ant-pos.locit.zarr"
    position_no_comb_name = "ant-pos-no-comb.position.zarr"
    position_simple_comb_name = "ant-pos-simple-comb.position.zarr"
    position_diff_comb_name = "ant-pos-diff-comb.position.zarr"

    position_files = {
        "no": f"{data_dir}/{position_no_comb_name}",
        "simple": f"{data_dir}/{position_simple_comb_name}",
        "difference": f"{data_dir}/{position_diff_comb_name}",
    }

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.phase_cal_table_name, folder=cls.data_dir)
        data.download(file=cls.ref_products_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

        extract_locit(cls.phase_cal_table_name, cls.locit_name, overwrite=True)

        for ddi_combination, pos_file_name in cls.position_files.items():
            locit(
                cls.locit_name,
                pos_file_name,
                combine_ddis=ddi_combination,
                overwrite=True,
            )

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir, ignore_errors=True)
            shutil.rmtree(cls.destination_folder, ignore_errors=True)
        return

    def test_position_mds_init(self):
        position_mds = AstrohackPositionFile(self.position_no_comb_name)
        assert isinstance(position_mds, AstrohackPositionFile)
        assert isinstance(position_mds, AstrohackLocitFile)

    def test_position_mds_text_exports(self):
        pos_res_name_dict = {
            "no": "position_separated_ddis_fit_results.txt",
            "simple": "position_combined_simple_fit_results.txt",
            "difference": "position_combined_difference_fit_results.txt",
        }
        for label, filename in self.position_files.items():
            position_mds = open_position(filename)

            fit_res_filename = pos_res_name_dict[label]
            position_mds.export_locit_fit_results(self.destination_folder)
            if not produce_reference_data():
                assert are_txt_files_equal(
                    f"{self.destination_folder}/{fit_res_filename}",
                    f"{self.ref_products_name}/{fit_res_filename}",
                ), f"{fit_res_filename} differs from reference file."

            parminator_filename = f"parminator_{label}_combination.par"
            position_mds.export_results_to_parminator(
                f"{self.destination_folder}/{parminator_filename}",
                correction_threshold=0.001,
                ddi=0,  # DDI specified for the no comb case
            )
            if not produce_reference_data():
                assert are_txt_files_equal(
                    f"{self.destination_folder}/{parminator_filename}",
                    f"{self.ref_products_name}/{parminator_filename}",
                ), f"{parminator_filename} differs from reference file."
        return

    @pytest.mark.skip(
        reason="Plot comparison is flaky and cannot be trusted to yield consistent results"
    )
    def test_position_mds_plot_exports(self):
        ddi = 0
        ant = "ea16"
        ant_pos_name_dict = {
            "no": "position_corrections_separated_ddi_0.png",
            "simple": "position_corrections_combined_simple.png",
            "difference": "position_corrections_combined_difference.png",
        }
        delay_name_dict = {
            "no": "position_delays_ant_ea16_separated_ddi_0.png",
            "simple": "position_delays_ant_ea16_combined_simple.png",
            "difference": "position_delays_ant_ea16_combined_difference.png",
        }
        sky_coverage_name_dict = {
            "no": "position_sky_coverage_ant_ea16_ddi_0.png",
            "simple": None,  # Simple and difference files are in thesis equal, but may differ in metadata, since they
            # have the same exact name, simple is set to None in order to avoid unreasonable test failures
            "difference": "position_sky_coverage_ant_ea16.png",
        }

        for label, filename in self.position_files.items():
            position_mds = open_position(filename)

            position_mds.plot_sky_coverage(self.destination_folder, ant=ant, ddi=ddi)
            if sky_coverage_name_dict[label] is not None:
                equal, msg = are_png_files_close(
                    f"{self.destination_folder}/{sky_coverage_name_dict[label]}",
                    f"{self.ref_products_name}/{sky_coverage_name_dict[label]}",
                )
                if not produce_reference_data():
                    assert equal, (
                        f"{msg}: {sky_coverage_name_dict[label]} differs from reference file."
                    )

            position_mds.plot_delays(self.destination_folder, ant=ant, ddi=ddi)
            if not produce_reference_data():
                equal, msg = are_png_files_close(
                    f"{self.destination_folder}/{delay_name_dict[label]}",
                    f"{self.ref_products_name}/{delay_name_dict[label]}",
                )
                assert equal, (
                    f"{msg}: {delay_name_dict[label]} differs from reference file."
                )

            position_mds.plot_position_corrections(self.destination_folder, ddi=ddi)
            if not produce_reference_data():
                equal, msg = are_png_files_close(
                    f"{self.destination_folder}/{ant_pos_name_dict[label]}",
                    f"{self.ref_products_name}/{ant_pos_name_dict[label]}",
                )
                assert equal, (
                    f"{msg}: {ant_pos_name_dict[label]} differs from reference file."
                )

        return

    def test_position_mds_structure(self):
        depth_dict = {
            "no": 2,
            "simple": 1,
            "difference": 1,
        }

        for label, filename in self.position_files.items():
            position_mds = open_position(filename)
            expected_depth = depth_dict[label]
            assert position_mds.root.depth == expected_depth, (
                f"{label.capitalize()} combination mds must have a depth of {expected_depth}."
            )
            assert "reference_antenna" in position_mds.root.attrs, (
                f"{label.capitalize()} combination mds must have a root attribute for reference antenna."
            )
            assert "telescope_name" in position_mds.root.attrs, (
                f"{label.capitalize()} combination mds must have a root attribute for telescopa name."
            )
