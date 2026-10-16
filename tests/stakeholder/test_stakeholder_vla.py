import json
import shutil

import numpy as np
import toolviper

from astrohack import open_image
from astrohack.core.holog_obs_dict import HologObsDict
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing
from astrohack.holog import holog
from astrohack.io.dio import open_panel
from astrohack.panel import panel
from astrohack.utils.conversion import convert_unit
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_dicts_close,
    execute_cleanup,
    produce_reference_data,
)


class TestStakeholder:
    data_dir = "stakeholder_test_data"

    ant = "ea25"
    ddi = 0
    ant_key = f"ant_{ant}"
    ddi_key = f"ddi_{ddi}"

    ms_prefix = "_cal_small_"
    ms_suffix = "_fixed.split.ms"

    epochs = ["before", "after"]

    ref_json_name = "vla_stakeholder_ref.json"
    ref_json_dict = {}
    extensions = {
        "pnt": "point.zarr",
        "hlg": "holog.zarr",
        "img": "image.zarr",
        "pnl": "panel.zarr",
    }

    # These values are in mils
    expected_panel_shifts = [-100, 75, 0, 150]
    shifted_panel_list = ["3-4", "5-27", "5-37", "5-38"]

    @classmethod
    def get_center_pixels(cls, epoch):
        img_name = cls.get_name(epoch, "img")

        img_xds = open_image(img_name)[cls.ant_key][cls.ddi_key]

        aperture_shape = (
            img_xds.APERTURE.values.shape[-2],
            img_xds.APERTURE.values.shape[-1],
        )
        beam_shape = img_xds.BEAM.values.shape[-2], img_xds.BEAM.values.shape[-1]

        aperture_center_pixels = np.squeeze(
            img_xds.APERTURE.values[..., aperture_shape[0] // 2, aperture_shape[1] // 2]
        )
        beam_center_pixels = np.squeeze(
            img_xds.BEAM.values[..., beam_shape[0] // 2, beam_shape[1] // 2]
        )

        return {
            "beam": {
                "real": beam_center_pixels.real.tolist(),
                "imag": beam_center_pixels.imag.tolist(),
            },
            "aperture": {
                "real": aperture_center_pixels.real.tolist(),
                "imag": aperture_center_pixels.imag.tolist(),
            },
        }

    @classmethod
    def get_panel_shifts(cls):
        m_to_mils = convert_unit("m", "mils", "length")
        corr_dict = {}
        for epoch in cls.epochs:
            pnl_name = cls.get_name(epoch, "pnl")
            pnl_xds = open_panel(pnl_name)[cls.ant_key][cls.ddi_key]

            corr_dict[epoch] = (
                pnl_xds.sel(labels=cls.shifted_panel_list).PANEL_SCREWS.values
                * m_to_mils
            )

        all_corr_shift = corr_dict["after"] - corr_dict["before"]
        mean_shift = np.mean(all_corr_shift, axis=1)
        return mean_shift

    @classmethod
    def get_name(cls, epoch, ext):
        return f"{cls.data_dir}/{cls.ant}_{epoch}{cls.extensions[ext]}"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""

        toolviper.utils.data.download(file=cls.ref_json_name, folder=cls.data_dir)

        for epoch in cls.epochs:
            ms_name = f"{cls.ant}{cls.ms_prefix}{epoch}{cls.ms_suffix}"
            pnt_name = cls.get_name(epoch, "pnt")
            hlg_name = cls.get_name(epoch, "hlg")
            img_name = cls.get_name(epoch, "img")
            pnl_name = cls.get_name(epoch, "pnl")

            toolviper.utils.data.download(file=ms_name, folder=cls.data_dir)

            ms_name = f"{cls.data_dir}/{ms_name}"
            extract_pointing(
                ms_name=ms_name,
                point_name=pnt_name,
                parallel=False,
                overwrite=True,
            )

            extract_holog(
                ms_name=ms_name,
                point_name=pnt_name,
                holog_name=hlg_name,
                data_column="CORRECTED_DATA",
                ant=cls.ant,
                ddi=cls.ddi,
                parallel=False,
                overwrite=True,
            )

            holog(
                holog_name=hlg_name,
                image_name=img_name,
                ant=cls.ant,
                ddi=cls.ddi,
                overwrite=True,
                parallel=False,
            )

            panel(
                image_name=img_name,
                panel_name=pnl_name,
                panel_model="flexible",
                ant=cls.ant,
                ddi=cls.ddi,
                parallel=False,
                overwrite=True,
            )

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir, ignore_errors=True)
        if produce_reference_data():
            with open(cls.ref_json_name, "w") as json_file:
                json.dump(cls.ref_json_dict, json_file)

    def test_holog_obs_dict(self):
        res_hlg_obs_dict = {}
        for epoch in self.epochs:
            hlg_name = self.get_name(epoch, "hlg")
            hlg_obs_dict = HologObsDict.from_holog_file(hlg_name)
            res_hlg_obs_dict[epoch] = hlg_obs_dict

        if produce_reference_data():
            self.ref_json_dict["holog_obs_dict"] = res_hlg_obs_dict

        else:
            with open(self.ref_json_name) as json_file:
                ref_dict = json.load(json_file)
            ref_hlg_obs_dict = ref_dict["holog_obs_dict"]

            assert are_dicts_close(res_hlg_obs_dict, ref_hlg_obs_dict), (
                "Reference and computed holog observation dicts do not match"
            )

    def test_center_pixels(self):
        res_pixels_dict = {}
        for epoch in self.epochs:
            res_pixels_dict[epoch] = self.get_center_pixels(epoch)

        if produce_reference_data():
            self.ref_json_dict["center_pixels"] = res_pixels_dict

        else:
            with open(self.ref_json_name) as json_file:
                ref_dict = json.load(json_file)
            ref_pixels_dict = ref_dict["center_pixels"]

            assert are_dicts_close(res_pixels_dict, ref_pixels_dict, tol=1e-6), (
                "Reference and computed center pixels do not match"
            )

        return

    def test_panel_shifts(self):
        res_mean_shift = self.get_panel_shifts()

        if produce_reference_data():
            self.ref_json_dict["panel_shifts"] = res_mean_shift.tolist()
        else:
            with open(self.ref_json_name) as json_file:
                ref_dict = json.load(json_file)
            ref_mean_shift = ref_dict["panel_shifts"]

            res_delta_from_true = res_mean_shift - self.expected_panel_shifts
            ref_delta_from_true = np.array(ref_mean_shift) - np.array(
                self.expected_panel_shifts
            )

            print("Result:", np.abs(res_delta_from_true))
            print("Reference:", np.abs(ref_delta_from_true))
            if (
                np.sum(np.abs(res_delta_from_true) < np.abs(ref_delta_from_true))
                > len(self.expected_panel_shifts) // 2
            ):
                evaluation = "better!"
            else:
                evaluation = "worse..."

            assert np.allclose(res_delta_from_true, ref_delta_from_true, atol=1e-6), (
                f"Panel shifts have changed and we are doing {evaluation}"
            )
