import shutil

import matplotlib
import numpy as np
from toolviper.utils import data

from astrohack import AstrohackPointFile, generate_holog_obs_dict, open_pointing
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_lists_equal,
    are_png_files_close,
    execute_cleanup,
    produce_reference_data,
)

matplotlib.use("Agg")


class TestPointMDS:
    data_dir = "point_data"
    destination_folder = "point_exports"
    ref_products_name = "ref_point_products"

    pnt_name = "ea25_cal_small_before_reference.point.zarr"

    ant_id = "ea25"
    ant_key = f"ant_{ant_id}"
    alt_ant_id = "ea06"
    alt_ant_key = f"ant_{alt_ant_id}"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.pnt_name, folder=cls.data_dir)
        data.download(file=cls.ref_products_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)
            shutil.rmtree(cls.destination_folder)
        return

    def test_init(self):
        pnt_mds = AstrohackPointFile(self.pnt_name)
        assert isinstance(pnt_mds, AstrohackPointFile)

    def test_plot_exports(self):
        pnt_mds = open_pointing(self.pnt_name)

        pnt_mds.plot_array_configuration(self.destination_folder)
        if not produce_reference_data():
            plot_name = "point_array_configuration.png"
            assert are_png_files_close(
                f"{self.destination_folder}/{plot_name}",
                f"{self.ref_products_name}/{plot_name}",
            ), "Array configuration plot is failing closeness test"

        pnt_mds.plot_pointing_in_time(
            self.destination_folder, plot_antennas_separately=False
        )
        if not produce_reference_data():
            plot_name = "point_directional_cosines_combined.png"
            assert are_png_files_close(
                f"{self.destination_folder}/{plot_name}",
                f"{self.ref_products_name}/{plot_name}",
            ), (
                "All antennas combined directional cosines plot is failing closeness test"
            )

        pnt_mds.plot_pointing_in_time(
            self.destination_folder,
            plot_antennas_separately=True,
            ant=self.ant_id,
        )
        if not produce_reference_data():
            plot_name = "point_directional_cosines_ant_ea25.png"
            assert are_png_files_close(
                f"{self.destination_folder}/{plot_name}",
                f"{self.ref_products_name}/{plot_name}",
            ), (
                "All antennas combined directional cosines plot is failing closeness test"
            )

    def compute_simple_hash(self, pnt_mds: AstrohackPointFile, key_to_hash):
        return np.sum(np.abs(pnt_mds[self.ant_key][key_to_hash].values))

    def test_set_antennas_as_reference(self):
        pnt_mds = open_pointing(self.pnt_name)

        old_alt_xdt = pnt_mds[self.alt_ant_key]

        old_dir_hash = self.compute_simple_hash(pnt_mds, "DIRECTION")
        old_tgt_hash = self.compute_simple_hash(pnt_mds, "TARGET")

        pnt_mds.set_antennas_as_reference(self.ant_id)

        new_dir_hash = self.compute_simple_hash(pnt_mds, "DIRECTION")
        new_tgt_hash = self.compute_simple_hash(pnt_mds, "TARGET")
        new_pnt_off_hash = self.compute_simple_hash(pnt_mds, "POINTING_OFFSET")
        new_dir_cos_hash = self.compute_simple_hash(pnt_mds, "DIRECTIONAL_COSINES")

        assert old_dir_hash == new_dir_hash, "Direction hash should not change"
        assert old_tgt_hash != new_tgt_hash, "Target hash should have changed"
        assert new_tgt_hash == new_dir_hash, (
            "Target and direction hashes should be equal"
        )
        assert new_pnt_off_hash == 0, "New pointing offset hash should be zero"
        assert new_dir_cos_hash == 0, "New directional cosines hash should be zero"

        new_alt_xdt = pnt_mds[self.alt_ant_key]

        assert new_alt_xdt == old_alt_xdt, "Other antenna Data tree should be unchanged"

        hlg_obs_dict = generate_holog_obs_dict(self.pnt_name)
        exp_ref_ants = ["ea04", self.ant_id]

        for ddi_key, ddi_dict in hlg_obs_dict.items():
            ant_dict = ddi_dict["map_0"]["ant"]
            ant_keys = list(ant_dict.keys())
            assert len(ant_keys) == 1, (
                f"There should be a single mapping antenna for {ddi_key}"
            )
            assert ant_keys[0] == self.alt_ant_id, (
                f"The only mapping antenna present for {ddi_key} should be {self.alt_ant_id}"
            )

            ref_ants = ant_dict[self.alt_ant_id]
            assert are_lists_equal(ref_ants, exp_ref_ants), (
                f"Reference antennas should be {exp_ref_ants} but got {ref_ants}"
            )
