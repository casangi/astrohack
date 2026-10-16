import os
import shutil

import matplotlib
import pytest
from toolviper.utils import data

from astrohack import AstrohackBeamcutFile, open_beamcut
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_png_files_close,
    are_txt_files_equal,
    execute_cleanup,
    produce_reference_data,
)

matplotlib.use("Agg")


class TestBeamcutMDS:
    data_dir = "beamcut_data"
    destination_folder = "beamcut_exports"
    ref_products_name = "ref_beamcut_products"

    silly_name = "Anything"
    remote_beamcut_name = "kband_beamcut_small.beamcut.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.remote_beamcut_name, folder=cls.data_dir)
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

    def test_beamcut_mds_init(self):
        beamcut_mds = AstrohackBeamcutFile(self.remote_beamcut_name)

        assert isinstance(beamcut_mds, AstrohackBeamcutFile)

    def test_beamcut_mds_observation_summary(self):
        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        obs_summary_reference_name = f"{self.ref_products_name}/obs_summary.txt"
        local_obs_summary = f"{self.destination_folder}/obs_summary.txt"

        os.makedirs(self.destination_folder, exist_ok=True)
        beamcut_mds.observation_summary(local_obs_summary)

        if not produce_reference_data():
            assert are_txt_files_equal(local_obs_summary, obs_summary_reference_name), (
                "Observation summary should be exactly equal to reference observation summary"
            )
            return

    @pytest.mark.skip(
        reason="Plot comparison is flaky and cannot be trusted to yield consistent results"
    )
    def test_beamcut_mds_plots(self):
        ant = "ea15"
        ddi = 0
        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        exec_dict = {
            "amplitude": beamcut_mds.plot_in_amplitude,
            "db": beamcut_mds.plot_in_tapering,
            "lm_offsets": beamcut_mds.plot_lm_offsets,
            "phase": beamcut_mds.plot_in_phase,
        }
        for plot_type, func in exec_dict.items():
            plot_filename = f"beamcut_{plot_type}_ant_{ant}_ddi_{ddi}.png"
            func(self.destination_folder, ant=ant, ddi=ddi)
            if not produce_reference_data():
                equal, msg = are_png_files_close(
                    f"{self.destination_folder}/{plot_filename}",
                    f"{self.ref_products_name}/{plot_filename}",
                )
                assert equal, (
                    f"{msg}: {plot_type} plot png file is different from the expected png file"
                )

    def test_beamcut_mds_fit_report(self):
        ant = "ea15"
        ddi = 0
        report_name = f"beamcut_report_ant_{ant}_ddi_{ddi}.txt"

        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        beamcut_mds.export_report(self.destination_folder, ant=ant, ddi=ddi)
        if not produce_reference_data():
            (
                are_txt_files_equal(
                    f"{self.destination_folder}/{report_name}",
                    f"{self.ref_products_name}/{report_name}",
                ),
                "Local and reference beamfit reports do not match",
            )
        return
