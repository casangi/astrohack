import pytest

import os
import json
import shutil
import toolviper

import numpy as np

from astrohack.holog import holog
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing


def relative_difference(result, expected):
    return 2 * np.abs(result - expected) / (abs(result) + abs(expected))


class TestHolog:
    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        toolviper.utils.data.download(
            file="ea25_cal_small_before_fixed.split.ms", folder="data/"
        )

        toolviper.utils.data.download(
            file="holog_numerical_verification.json", folder="data/"
        )

        extract_pointing(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            overwrite=True,
            parallel=False,
        )

        # Extract holography data using holog_obd_dict
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True,
        )

        extract_holog(
            ms_name="data/ea25_cal_small_before_fixed.split.ms",
            point_name="data/ea25_cal_small_before_fixed.split.point.zarr",
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True,
        )

        holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            grid_size=[31, 31],
            cell_size=[-0.0006386556122807017, 0.0006386556122807017],
            overwrite=True,
            parallel=False,
        )

        with open(
            "data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr"
        ) as json_attr:
            cls.json_file = json.load(json_attr)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree("data")

    def setup_method(self):
        """setup any state specific to all methods of the given class"""
        pass

    def teardown_method(self):
        """teardown any state that was previously setup for all methods of the given class"""
        pass

    def test_holog_grid_cell_size(self):
        """
        Calculate the correct grid and cell size when compared to known values in the test file; known values are
        provided by a test json file.
        """

        tolerance = 2.0e-5

        with open("data/holog_numerical_verification.json") as file:
            reference_dict = json.load(file)

        with open(
            "data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr"
        ) as attr_file:
            image_attr = json.load(attr_file)

        for i, _ in enumerate(image_attr["cell_size"]):
            print(image_attr["grid_size"][i])
            print(reference_dict["vla"]["grid_size"][i])

            assert (
                relative_difference(
                    image_attr["cell_size"][i], reference_dict["vla"]["cell_size"][i]
                )
                < tolerance
            )

            assert (
                relative_difference(
                    image_attr["grid_size"][i], reference_dict["vla"]["grid_size"][i]
                )
                < tolerance
            )

    def test_holog_image_name(self):
        """
        Test holog image name created correctly.
        """

        assert os.path.exists("data/ea25_cal_small_before_fixed.split.image.zarr")

    def test_holog_ant_id(self):
        """
        Specify a single antenna to process; check that is the only antenna returned.
        """

        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            ant=["ea25"],
            overwrite=True,
            parallel=False,
        )

        assert list(image_mds.keys()) == ["ant_ea25"]

    def test_holog_ddi(self):
        """
        Specify a single ddi to process; check that is the only ddi returned.
        """

        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            overwrite=True,
            ddi=[0],
            parallel=False,
        )

        for ant in image_mds.keys():
            for ddi in image_mds[ant].keys():
                assert ddi == "ddi_0"

    def test_holog_padding_factor(self):
        """
        Specify a padding factor to use in the image creation; check that image size is created.
        """

        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            padding_factor=10,
            overwrite=True,
            parallel=False,
        )

        for ant in image_mds.keys():
            for ddi in image_mds[ant].keys():
                assert image_mds[ant][ddi].APERTURE.shape == (1, 1, 4, 512, 512)

    def test_holog_chan_average(self):
        """
        Check that channel average flag was set holog is run.
        """
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            chan_average=True,
            overwrite=True,
            parallel=False,
        )

        with open(
            "data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr"
        ) as json_attr:
            json_file = json.load(json_attr)

        assert json_file["chan_average"] is True

    def test_holog_scan_average(self):
        """
        Check that scan average flag was set holog is run.
        """
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            scan_average=False,
            overwrite=True,
            parallel=False,
        )

        with open(
            "data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr"
        ) as json_attr:
            json_file = json.load(json_attr)

        assert not json_file["scan_average"]

    def test_holog_grid_interpolation(self):
        """
        Check that grid interpolation flag was set holog is run.
        """
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            grid_interpolation_mode="nearest",
            overwrite=True,
            parallel=False,
        )

        with open(
            "data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr"
        ) as json_attr:
            json_file = json.load(json_attr)

        assert json_file["grid_interpolation_mode"] == "nearest"

    def test_holog_chan_tolerance(self):
        """
        Check that channel tolerance is propagated correctly.
        """
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            chan_tolerance_factor=0.0049,
            overwrite=True,
            parallel=False,
        )

        with open(
            "data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr"
        ) as json_attr:
            json_file = json.load(json_attr)

        assert json_file["chan_tolerance_factor"] == 0.0049

    def test_holog_to_stokes(self):
        """
        Check that to_stokes flag was set holog is run.
        """
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            to_stokes=True,
            overwrite=True,
            parallel=False,
        )

        with open(
            "data/ea25_cal_small_before_fixed.split.image.zarr/.image_attr"
        ) as json_attr:
            json_file = json.load(json_attr)

        assert json_file["to_stokes"]

        assert (
            image_mds["ant_ea25"]["ddi_0"].pol.values == np.array(["I", "Q", "U", "V"])
        ).all()

    def test_holog_overwrite(self):
        """
        Specify the output file should be overwritten; check that it WAS.
        """
        initial_time = os.path.getctime(
            "data/ea25_cal_small_before_fixed.split.image.zarr"
        )

        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            overwrite=True,
            parallel=False,
        )

        modified_time = os.path.getctime(
            "data/ea25_cal_small_before_fixed.split.image.zarr"
        )

        assert initial_time != modified_time

    def test_holog_perturbation_phase_fit(self):
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            phase_fit_engine="perturbations",
            grid_size=[31, 31],
            cell_size=[-0.0006386556122807017, 0.0006386556122807017],
            overwrite=True,
            parallel=False,
        )
        keys = [
            "phase_offset",
            "x_cassegrain_offset",
            "x_focus_offset",
            "x_point_offset",
            "x_subreflector_tilt",
            "y_cassegrain_offset",
            "y_focus_offset",
            "y_point_offset",
            "y_subreflector_tilt",
            "z_focus_offset",
        ]
        references = [
            0.07578374993954257,
            -28.033780511487777,
            -1.9620592050595538,
            0.00016673100624246893,
            0.00036714280075938257,
            -22.752401475110595,
            -3.3596837733268057,
            0.00032344494918384674,
            -0.0006101436903899218,
            0.07222802059408939,
        ]

        pha_fit_res = image_mds["ant_ea25"]["ddi_0"].attrs["phase_fitting"]["map_0"][
            "14167000000.0"
        ]["I"]

        for ikey, key in enumerate(keys):
            assert np.isclose(pha_fit_res[key]["value"], references[ikey]), (
                f"Phase fitting values differ from " f"reference for {key}"
            )

        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            phase_fit_engine="perturbations",
            phase_fit_control=[False, False, False, False, False],
            overwrite=True,
            parallel=False,
        )
        pha_fit_res = image_mds["ant_ea25"]["ddi_0"].attrs["phase_fitting"]

        assert pha_fit_res is None

        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            phase_fit_engine="perturbations",
            phase_fit_control=[False, True, False, True, False],
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds["ant_ea25"]["ddi_0"].attrs["phase_fitting"]["map_0"][
            "14167000000.0"
        ]["I"]

        assert np.isnan(pha_fit_res["x_point_offset"]["error"])
        assert np.isnan(pha_fit_res["z_focus_offset"]["error"])
        assert np.isnan(pha_fit_res["x_cassegrain_offset"]["error"])

    def test_holog_no_phase_fitting(self):
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            phase_fit_engine="none",
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds["ant_ea25"]["ddi_0"].attrs["phase_fitting"]

        assert pha_fit_res is None

    def test_holog_zernike_phase_fitting(self):
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            grid_size=[31, 31],
            cell_size=[-0.0006386556122807017, 0.0006386556122807017],
            phase_fit_engine="zernike",
            zernike_n_order=4,
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds["ant_ea25"]["ddi_0"].attrs["phase_fitting"]

        assert pha_fit_res is None

        ref_phase = [
            [[125, 125], -0.17758619948993593],
            [[213, 430], -0.1459607430199923],
            [[432, 195], -0.034865251933011265],
        ]
        phase_img = image_mds["ant_ea25"]["ddi_0"].CORRECTED_PHASE.values[0, 0, 0]

        for idx, phase in ref_phase:
            assert np.isclose(
                phase_img[*idx], phase
            ), f"Phase is different from reference at {idx}"

    def test_holog_zernike_coeffs(self):
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            grid_size=[31, 31],
            cell_size=[-0.0006386556122807017, 0.0006386556122807017],
            phase_fit_engine="none",
            zernike_n_order=10,
            overwrite=True,
            parallel=False,
        )
        ref_zernike_coeffs = [
            3.63853142e-01 + 3.22356554e-02j,
            4.23001476e00 + 4.11000920e01j,
            -1.75638037e01 - 6.11203902e01j,
            5.44624567e00 - 2.71889063e00j,
            -2.07518837e-01 - 1.07029883e-02j,
            -5.32112029e00 + 4.64863302e00j,
            -4.44881637e00 - 4.13055928e01j,
            4.26979653e00 + 3.99186278e01j,
            -8.98318910e00 - 3.19533343e01j,
            1.14296814e01 + 4.01677823e01j,
            3.22839441e00 - 1.57685943e00j,
            3.50888476e00 - 1.55562367e00j,
            -3.57103888e-02 + 1.19658639e-02j,
            -2.27098458e-02 + 2.00348932e-02j,
            9.35219135e-03 - 1.41071395e-02j,
            1.83504535e00 + 1.73453819e01j,
            -1.95266260e00 - 1.77442134e01j,
            1.85819383e00 + 1.71604139e01j,
            -1.83994376e-01 - 7.02351715e-01j,
            -1.17611308e00 - 4.27534468e00j,
            1.07348587e00 + 3.73876032e00j,
            2.45203816e00 - 1.12858846e00j,
            3.60288701e-01 - 2.00945408e-01j,
            7.99358267e-01 - 3.78177411e-01j,
            6.87383554e-04 - 3.05145445e-02j,
            -6.70306575e-02 + 2.45662583e-03j,
            2.51829138e-02 + 2.42719024e-03j,
            1.51297987e-03 - 2.09583627e-02j,
            -2.18272416e-01 - 2.49254874e00j,
            3.18745948e-01 + 2.90377897e00j,
            -3.48360089e-01 - 2.98705808e00j,
            2.82767637e-01 + 2.65015144e00j,
            -6.45378731e-02 - 2.32601543e-01j,
            -1.78847682e-01 - 6.28803632e-01j,
            1.98984840e-01 + 6.45562396e-01j,
            -5.75574487e-01 - 2.03482119e00j,
            -1.79071119e-01 + 9.75425501e-02j,
            1.33041282e-01 - 6.08107273e-02j,
            -1.04930214e-01 + 1.98600004e-02j,
            1.02529432e-01 - 4.56826111e-02j,
            -9.10502057e-02 + 4.00437485e-02j,
            -4.55895027e-02 + 3.88348143e-02j,
            4.98863504e-03 - 2.49634561e-02j,
            1.81918402e-02 + 1.18122844e-02j,
            -1.60219775e-02 - 2.28458691e-02j,
            1.47953573e-02 + 1.31553475e-01j,
            7.77135087e-03 + 3.36702956e-02j,
            2.47718592e-02 - 3.50457522e-02j,
            -2.67166187e-02 + 1.18180934e-02j,
            -1.46191264e-02 - 7.72276561e-02j,
            -3.00941394e-02 - 9.87108070e-02j,
            8.96918830e-03 - 2.84022568e-02j,
            -1.65357311e-03 - 4.38318775e-02j,
            1.50658709e-02 - 6.14483136e-02j,
            4.87811290e-02 + 1.75694507e-01j,
            -4.81118972e-03 + 3.73321157e-03j,
            4.60629264e-02 - 3.62603959e-02j,
            -1.21179544e-03 + 2.53104896e-02j,
            1.75069126e-02 + 1.04703864e-02j,
            8.61753013e-03 + 2.54347275e-03j,
            5.40815261e-02 - 2.62438937e-02j,
            4.45687952e-02 - 4.29741834e-02j,
            4.25176984e-02 + 4.59819354e-02j,
            -1.46957058e-02 - 3.46945617e-02j,
            3.38637874e-02 - 1.08213929e-02j,
            -2.06739538e-02 + 1.35215571e-03j,
        ]

        zer_coeffs = image_mds["ant_ea25"]["ddi_0"].ZERNIKE_COEFFICIENTS.values[0, 0, 0]

        expected_n_coeff = 66
        assert zer_coeffs.shape[0] == expected_n_coeff

        assert np.allclose(
            ref_zernike_coeffs, zer_coeffs
        ), "Fitted Zernike coefficients do not match references"

    def test_holog_not_overwrite(self):
        """
        Specify the output file should be NOT be overwritten; check that it WAS NOT.
        """
        initial_time = os.path.getctime(
            "data/ea25_cal_small_before_fixed.split.image.zarr"
        )

        try:
            holog(
                holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
                image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
                overwrite=False,
                parallel=False,
            )

        except FileExistsError:
            pass

        finally:
            modified_time = os.path.getctime(
                "data/ea25_cal_small_before_fixed.split.image.zarr"
            )

            assert initial_time == modified_time
