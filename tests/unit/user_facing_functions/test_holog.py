import json
import os
import pathlib
import shutil

import numpy as np
import pytest
import toolviper

from astrohack import open_image
from astrohack.holog import holog
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    execute_cleanup,
    produce_reference_data,
)


class TestHolog:
    data_dir = "holog_data"
    hlg_name = "ea25_cal_small_before_reference.holog.zarr"

    def_img_name = "ea25_cal_small_before_reference.image.zarr"
    ref_img_name = "ea25_cal_before_reference.image.zarr"
    phase_fit_img_name = "phase_fit.image.zarr"
    zern_pha_fit_name = "zern_pha.image.zarr"
    zern_coeff_name = "zern_coeff.image.zarr"

    ant_id = "ea25"
    ddi_id = 0
    ant_key = f"ant_{ant_id}"
    ddi_key = f"ddi_{ddi_id}"

    ref_json_name = "holog-ref-values.json"
    ref_json_dict = {}

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        toolviper.utils.data.download(file=cls.hlg_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_img_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_json_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)
        if produce_reference_data():
            with open(cls.ref_json_name, "w", encoding="utf-8") as json_file:
                json.dump(cls.ref_json_dict, json_file, indent=4)

    def test_defaults(self):
        """
        test holog defaults
        """
        new_img_mds = holog(
            holog_name=self.hlg_name,
            overwrite=True,
        )

        assert pathlib.Path(self.def_img_name).is_dir(), (
            f"A .image.zarr file named {self.def_img_name} does not exist."
        )
        if produce_reference_data():
            return

        ref_img_mds = open_image(self.ref_img_name)
        assert new_img_mds.is_close_to(ref_img_mds), (
            "Reference and new mdses are different."
        )

    def test_data_selection(self):
        if produce_reference_data():
            return
        image_mds = holog(
            holog_name=self.hlg_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
        )

        ant_list = list(image_mds.keys())
        assert len(ant_list) == 1, "A single antenna should be present."
        assert ant_list[0] == self.ant_key, (
            "Ant name should be the same as the one given."
        )

        ddi_list = list(image_mds[self.ant_key].keys())
        assert len(ddi_list) == 1, "A single ddi should be present."
        assert ddi_list[0] == self.ddi_key, (
            "DDI key should be the same as the one given."
        )

    def test_padding_factor(self):
        """
        Specify a padding factor to use in the image creation; check that image size is created.
        """
        if produce_reference_data():
            return
        pad_list = [[5, 256], [10, 512]]

        for pad_fac, ap_size in pad_list:
            image_mds = holog(
                holog_name=self.hlg_name,
                padding_factor=pad_fac,
                ant=self.ant_id,
                ddi=self.ddi_id,
                overwrite=True,
                parallel=False,
            )
            ap_shape = (
                1,
                1,
                4,
                ap_size,
                ap_size,
            )
            for ant_key in image_mds.keys():
                for ddi_key in image_mds[ant_key].keys():
                    this_ap_shape = image_mds[ant_key][ddi_key].APERTURE.shape
                    assert this_ap_shape == ap_shape, (
                        f"Aperture for {ant_key} {ddi_key} for a padding factor of {pad_fac} should be {ap_shape} "
                        f"but is {this_ap_shape}."
                    )

    def test_chan_average(self):
        """
        Check that channel average flag was set holog is run.
        """
        if produce_reference_data():
            return
        ref_nchan = 1
        image_mds = holog(
            holog_name=self.hlg_name,
            chan_average=True,
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
            parallel=False,
        )

        for ant_key in image_mds.keys():
            for ddi_key in image_mds[ant_key].keys():
                this_ap_shape = image_mds[ant_key][ddi_key].APERTURE.shape
                assert this_ap_shape[1] == ref_nchan, (
                    f"Non chan_average aperture for {ant_key} {ddi_key} should have {ref_nchan} channels"
                )

    def test_to_stokes(self):
        """
        Check that to_stokes flag was set holog is run.
        """
        if produce_reference_data():
            return
        stokes_axis = np.array(["I", "Q", "U", "V"])
        image_mds = holog(
            holog_name=self.hlg_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            to_stokes=True,
            overwrite=True,
            parallel=False,
        )
        pol_axis = image_mds[self.ant_key][self.ddi_key].pol.values
        assert np.all(pol_axis == stokes_axis), (
            f"If to_stokes is set to True output data should have {stokes_axis} as the polarization axis."
        )

    def test_overwrite(self):
        """
        Specify the output file should be overwritten; check that it WAS.
        """
        if produce_reference_data():
            return
        initial_time = os.path.getctime(self.def_img_name)

        holog(
            holog_name=self.hlg_name,
            image_name=self.def_img_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
            parallel=False,
        )
        modified_time = os.path.getctime(self.def_img_name)
        assert initial_time != modified_time, (
            "Recreated file has to have a different time from the original file."
        )

        with pytest.raises(FileExistsError):
            holog(
                holog_name=self.hlg_name,
                image_name=self.def_img_name,
                ant=self.ant_id,
                ddi=self.ddi_id,
                overwrite=False,
                parallel=False,
            )

    def test_perturbation_phase_fit(self):
        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.phase_fit_img_name,
            phase_fit_engine="perturbations",
            ant=self.ant_id,
            ddi=self.ddi_id,
            grid_size=[31, 31],
            cell_size=[-0.0006386556122807017, 0.0006386556122807017],
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"][
            "map_0"
        ]["14167000000.0"]["I"]

        if produce_reference_data():
            self.ref_json_dict["phase_fit_reference"] = pha_fit_res
            return

        with open(self.ref_json_name) as json_file:
            ref_dict = json.load(json_file)

        ref_phase_fit = ref_dict["phase_fit_reference"]
        for key in ref_phase_fit.keys():
            assert np.isclose(pha_fit_res[key]["value"], ref_phase_fit[key]["value"]), (
                f"Phase fitting values differ from reference for {key}"
            )

        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.phase_fit_img_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            phase_fit_engine="perturbations",
            phase_fit_control=[False, False, False, False, False],
            overwrite=True,
            parallel=False,
        )
        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"]

        assert pha_fit_res is None, (
            "When phase_fit_control is a 5-way False tuple, phase fit results should be None"
        )

        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.phase_fit_img_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            phase_fit_engine="perturbations",
            phase_fit_control=[False, True, False, True, False],
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"][
            "map_0"
        ]["14167000000.0"]["I"]

        assert np.isnan(pha_fit_res["x_point_offset"]["error"]), (
            "If pointing offset is not fitted x_point_offset error should be NaN"
        )
        assert np.isnan(pha_fit_res["z_focus_offset"]["error"]), (
            "If focus is not fitted z_focus_offset error should be NaN"
        )
        assert np.isnan(pha_fit_res["x_cassegrain_offset"]["error"]), (
            "If cassegrain offset is not fitted x_cassegrain_offset error should be NaN"
        )

    def test_no_phase_fit(self):
        if produce_reference_data():
            return
        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.def_img_name,
            phase_fit_engine="none",
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"]
        assert pha_fit_res is None, (
            "When phase_fit_engine is set to 'none', phase fit results should be None"
        )

    def test_zernike_phase_fitting(self):
        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.zern_pha_fit_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            phase_fit_engine="zernike",
            zernike_n_order=4,
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"]

        assert pha_fit_res is None
        positions = [[125, 125], [213, 430], [432, 195], [125, 309], [432, 203]]

        phase_img = image_mds[self.ant_key][self.ddi_key].CORRECTED_PHASE.values[
            0, 0, 0
        ]
        if produce_reference_data():
            corrected_phase_dict = {}
            for i_key, position in enumerate(positions):
                corrected_phase_dict[i_key] = phase_img[*position]
            self.ref_json_dict["corrected_phase_ref"] = corrected_phase_dict
            return

        with open(self.ref_json_name) as json_file:
            ref_dict = json.load(json_file)

        corrected_phase_dict = ref_dict["corrected_phase_ref"]
        for i_key, phase in corrected_phase_dict.items():
            position = positions[int(i_key)]
            assert np.isclose(phase_img[*position], phase), (
                f"Phase is different from reference at {position}"
            )

    #
    def test_holog_zernike_coeffs(self):
        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.zern_coeff_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            phase_fit_engine="none",
            zernike_n_order=10,
            overwrite=True,
            parallel=False,
        )

        zer_coeffs = image_mds[self.ant_key][self.ddi_key].ZERNIKE_COEFFICIENTS.values[
            0, 0, 0
        ]

        expected_n_coeff = 66
        assert zer_coeffs.shape[0] == expected_n_coeff

        if produce_reference_data():
            self.ref_json_dict["zernike_coeff_ref_real"] = zer_coeffs.real.tolist()
            self.ref_json_dict["zernike_coeff_ref_imag"] = zer_coeffs.imag.tolist()
            return

        with open(self.ref_json_name) as json_file:
            ref_dict = json.load(json_file)
        ref_zernike_coeffs_real = np.array(ref_dict["zernike_coeff_ref_real"])
        ref_zernike_coeffs_imag = np.array(ref_dict["zernike_coeff_ref_imag"])
        assert np.allclose(ref_zernike_coeffs_real, zer_coeffs.real), (
            "Fitted real part of Zernike coefficients do not match references"
        )
        assert np.allclose(ref_zernike_coeffs_imag, zer_coeffs.imag), (
            "Fitted imag part of Zernike coefficients do not match references"
        )
