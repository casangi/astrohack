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

        assert json_file["scan_average"] == False

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

        assert json_file["to_stokes"] == True

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
            0.07854467856879538,
            -28.044142593327745,
            -1.9400764990642743,
            0.0001641826578867972,
            0.0003639636675771953,
            -22.79834097424694,
            -3.316538216333087,
            0.00031687826284370707,
            -0.0006533042611128819,
            0.07242809618691809,
        ]

        pha_fit_res = image_mds["ant_ea25"]["ddi_0"].attrs["phase_fitting"]["map_0"][
            "14167000000.0"
        ]["I"]

        for ikey, key in enumerate(keys):
            assert np.abs(pha_fit_res[key]["value"] - references[ikey]) < 1e-6

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
            phase_fit_engine="zernike",
            zernike_n_order=4,
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds["ant_ea25"]["ddi_0"].attrs["phase_fitting"]

        assert pha_fit_res is None

        ref_idx = [[125, 125], [213, 430], [432, 195]]
        ref_phase = [-0.1832512763834142, -0.1476201796080896, 0.0009533632210843024]

        phase_img = image_mds["ant_ea25"]["ddi_0"].CORRECTED_PHASE.values[0, 0, 0]

        for itest in range(len(ref_phase)):
            ix, iy = ref_idx[itest]
            assert np.abs(phase_img[ix, iy] - ref_phase[itest]) < 1e-6

    def test_holog_zernike_coeffs(self):
        image_mds = holog(
            holog_name="data/ea25_cal_small_before_fixed.split.holog.zarr",
            image_name="data/ea25_cal_small_before_fixed.split.image.zarr",
            phase_fit_engine="none",
            zernike_n_order=10,
            overwrite=True,
            parallel=False,
        )
        ref_real_coeffs = [
            3.67692385e-01,
            4.04608169e00,
            -1.90211689e01,
            5.25906809e00,
            -2.11363263e-01,
            -5.01066365e00,
            -4.25080307e00,
            4.08715342e00,
            -9.75965238e00,
            1.23387440e01,
            3.11527206e00,
            3.40312731e00,
            -3.76809075e-02,
            -2.14798047e-02,
            1.12252909e-02,
            1.75536454e00,
            -1.86647611e00,
            1.78146568e00,
            -2.25846856e-01,
            -1.26767701e00,
            1.11794975e00,
            2.37608666e00,
            3.43735372e-01,
            7.73122809e-01,
            -7.43784007e-04,
            -6.63112706e-02,
            2.99250607e-02,
            3.86467383e-03,
            -2.12123976e-01,
            3.05646382e-01,
            -3.32851086e-01,
            2.70462177e-01,
            -7.27156610e-02,
            -1.90286233e-01,
            2.08162186e-01,
            -6.25319970e-01,
            -1.72975109e-01,
            1.30162463e-01,
            -1.05514408e-01,
            9.96154902e-02,
            -9.44357113e-02,
            -4.30239065e-02,
            6.70597104e-03,
            2.09167628e-02,
            -1.38605407e-02,
            1.24657251e-02,
            7.61012582e-03,
            2.47827441e-02,
            -2.60832526e-02,
            -1.34770793e-02,
            -3.37073939e-02,
            1.03922963e-02,
            -1.17801224e-03,
            1.43763850e-02,
            5.21020202e-02,
            -4.49054476e-03,
            4.70003748e-02,
            -9.23013783e-04,
            1.36310369e-02,
            4.33987340e-03,
            5.71422858e-02,
            4.79594076e-02,
            4.45392997e-02,
            -1.27793372e-02,
            3.78887289e-02,
            -1.71322356e-02,
        ]

        ref_imag_coeffs = [
            3.14448214e-02,
            4.01048404e01,
            -5.82465104e01,
            -2.83908318e00,
            -1.09321007e-02,
            4.20014988e00,
            -4.02587720e01,
            3.89401251e01,
            -3.04791533e01,
            3.82806616e01,
            -1.65193462e00,
            -1.62686785e00,
            1.04152859e-02,
            1.83789059e-02,
            -1.55332135e-02,
            1.69116423e01,
            -1.72934296e01,
            1.67433598e01,
            -6.73704910e-01,
            -4.07536804e00,
            3.54182587e00,
            -1.17878830e00,
            -2.10299603e-01,
            -3.95538254e-01,
            -3.15309775e-02,
            -3.74881316e-04,
            -1.50306058e-04,
            -1.81892073e-02,
            -2.44530164e00,
            2.82870933e00,
            -2.91077835e00,
            2.58461341e00,
            -2.22660652e-01,
            -5.97746735e-01,
            6.13322032e-01,
            -1.93814174e00,
            1.02131360e-01,
            -6.32883106e-02,
            2.25429158e-02,
            -4.75309530e-02,
            3.57927681e-02,
            3.50993973e-02,
            -2.99220945e-02,
            1.33202450e-02,
            -2.31228892e-02,
            1.24166566e-01,
            2.98858453e-02,
            -4.03686358e-02,
            1.11459648e-02,
            -7.23297543e-02,
            -9.34552632e-02,
            -2.69289500e-02,
            -4.66305199e-02,
            -6.09032890e-02,
            1.65926624e-01,
            3.55785778e-03,
            -3.55639233e-02,
            2.56953660e-02,
            1.06170022e-02,
            9.98199545e-04,
            -2.81930752e-02,
            -4.64636180e-02,
            3.83886557e-02,
            -3.22192938e-02,
            -1.00094676e-02,
            3.13298786e-03,
        ]

        zer_coeffs = image_mds["ant_ea25"]["ddi_0"].ZERNIKE_COEFFICIENTS.values[0, 0, 0]

        expected_n_coeff = 66
        assert zer_coeffs.shape[0] == expected_n_coeff

        for icoeff in range(expected_n_coeff):
            assert np.abs(zer_coeffs[icoeff].real - ref_real_coeffs[icoeff]) < 1e-6
            assert np.abs(zer_coeffs[icoeff].imag - ref_imag_coeffs[icoeff]) < 1e-6

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
