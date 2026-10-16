import shutil

import numpy as np
import toolviper

from astrohack import get_proper_telescope, open_holog
from astrohack.utils.constants import clight
from astrohack.utils.gridding import grid_1d_data, grid_beam, gridding_correction
from astrohack.utils.ray_tracing_general import simple_axis
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_dicts_close,
)


class TestGridAlgorithms:
    chan_tolerance_factor = 0.005
    vla = get_proper_telescope("vla")
    data_dir = "grid_data"

    hlg_name = "ea25_cal_small_before_reference.holog.zarr"

    ant_id = "ea25"
    ant_key = f"ant_{ant_id}"
    ddi_id = 0
    ddi_key = f"ddi_{ddi_id}"
    map_key = "map_0"

    beam_avg_shape = [1, 1, 4, 28, 28]
    beam_no_avg_shape = [1, 64, 4, 28, 28]

    fill_value = 2.0

    orig_x_axis = np.arange(0, 1000, 0.05)
    dest_x_under = np.arange(0, 1000, 0.1)
    dest_x_over = np.arange(0, 1000, 0.025)

    orig_y_data = [np.full([orig_x_axis.shape[0], 1], fill_value)]

    def resample_nan_count(self, method, fallback, exp_n_nan):
        local_orig_x_axis = np.arange(0, 1000, 0.05)
        local_orig_x_axis[5000:] += 0.34
        resamp_under = grid_1d_data(
            self.dest_x_under,
            local_orig_x_axis,
            self.orig_y_data,
            method,
            "test origin",
            "test under",
            gaussian_fallback=fallback,
            return_weights=False,
            second_dim_len=1,
        )

        n_nan = np.sum(np.isnan(resamp_under))
        assert n_nan == exp_n_nan, (
            "Number of NaNs is not what is expected when introducing an irregularity in the origin axis"
        )

    def resample_weight_test(self, method, is_over, ref_weights):
        if is_over:
            dest_x_axis = self.dest_x_over
            dest_label = f"test over {method}"
            mode = "over"

        else:
            dest_x_axis = self.dest_x_under
            dest_label = f"test under {method}"
            mode = "under"

        resamp_y, weights = grid_1d_data(
            dest_x_axis,
            self.orig_x_axis,
            self.orig_y_data,
            method,
            f"test origin {method}",
            dest_label,
            gaussian_fallback=False,
            return_weights=True,
            second_dim_len=1,
        )

        assert np.allclose(resamp_y, self.fill_value), (
            f"{method.capitalize()} {mode} resampled data does not have the expected values"
        )

        assert np.allclose(np.unique(weights), ref_weights), (
            f"{method.capitalize()} {mode} weights are different from what is expected"
        )

    def beam_grid_assertions(
        self,
        expected_shape,
        ref_center_values,
        ref_freq_axis,
        mode,
        grid_beam_tuple_return,
    ):
        (
            beam_grid,
            time_centroid,
            output_freq_axis,
            pol_axis,
            l_axis,
            m_axis,
            grid_corr,
            obs_sum,
        ) = grid_beam_tuple_return

        n_chan = expected_shape[1]
        i_chan = n_chan // 2
        i_x_cen = expected_shape[3] // 2
        i_y_cen = expected_shape[4] // 2

        assert np.all(np.isclose(expected_shape, beam_grid.shape)), (
            f"{mode.capitalize()} beam grid does not have the expected shape"
        )

        for i_pol, pol in enumerate(pol_axis):
            assert np.isclose(
                beam_grid[0, i_chan, i_pol, i_x_cen, i_y_cen], ref_center_values[i_pol]
            ), f"{mode.capitalize()} center pixel for {pol} does not match reference"

        assert np.isclose(time_centroid[0], self.expected_time_centroid), (
            f"{mode.capitalize()} time centroid is different from the expected"
        )

        assert l_axis.shape[0] == self.grid_size[0], (
            f"{mode.capitalize()} grid size and l axis size are not equal"
        )

        assert m_axis.shape[0] == self.grid_size[1], (
            f"{mode.capitalize()} grid size and m axis size are not equal"
        )

        assert np.all(np.isclose(ref_freq_axis, output_freq_axis)), (
            f"{mode.capitalize()} output frequency axis is not equal to the reference"
        )

        if "linear" in mode:
            assert not grid_corr, (
                f"{mode.capitalize()} beam grid does not warrants a gridding correction in aperture plane"
            )
        elif "gaussian" in mode:
            assert grid_corr, (
                f"{mode.capitalize()} beam grid warrants a gridding correction in aperture plane"
            )
        else:
            raise RuntimeError(f"Unrecognized mode {mode}")

        assert are_dicts_close(obs_sum, self.obs_sum, tol=1e-6), (
            f"{mode.capitalize()} observation summary differs from the expected"
        )

    @classmethod
    def setup_class(cls):
        toolviper.utils.data.download(cls.hlg_name, cls.data_dir)

        add_data_folder_to_names_in_class(cls)

        cls.holog_mds = open_holog(cls.hlg_name)

        cls.ant_ddi_xdt = cls.holog_mds[cls.ant_key][cls.ddi_key]

        cls.obs_sum = cls.ant_ddi_xdt["map_0"].attrs["summary"]
        cls.cell_size = np.array(
            [-cls.obs_sum["beam"]["cell size"], cls.obs_sum["beam"]["cell size"]]
        )
        cls.grid_size = np.array(cls.obs_sum["beam"]["grid size"])
        cls.expected_time_centroid = 5.16975892e09

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_dir)

    def test_grid_linear_beam_no_chan_average(self):
        mode = "linear no average"
        return_tuple = grid_beam(
            ant_ddi_xdt=self.ant_ddi_xdt,
            grid_size=self.grid_size,
            sky_cell_size=self.cell_size,
            avg_chan=False,
            telescope=self.vla,
            chan_tol_fac=self.chan_tolerance_factor,
            grid_interpolation_mode="linear",
            observation_summary=self.obs_sum,
            label=f"Test {mode}",
        )

        ref_freq_axis = self.ant_ddi_xdt[self.map_key].chan.values
        ref_center_values = [
            0.2780216888736269 + 0.0029742640382882136j,
            0.005223369026628943 + 0.018311068461403537j,
            -0.007586815158936162 + 0.006465368831061344j,
            0.2863594496065688 + 0.015925966825276847j,
        ]
        self.beam_grid_assertions(
            self.beam_no_avg_shape, ref_center_values, ref_freq_axis, mode, return_tuple
        )

    def test_grid_linear_beam_chan_average(self):
        mode = "linear average"
        return_tuple = grid_beam(
            ant_ddi_xdt=self.ant_ddi_xdt,
            grid_size=self.grid_size,
            sky_cell_size=self.cell_size,
            avg_chan=True,
            telescope=self.vla,
            chan_tol_fac=self.chan_tolerance_factor,
            grid_interpolation_mode="linear",
            observation_summary=self.obs_sum,
            label=f"Test {mode}",
        )

        ref_freq_axis = np.average(self.ant_ddi_xdt[self.map_key].chan.values)
        ref_center_values = [
            0.2555271590565431 + 0.013577819885576325j,
            0.006690232199874844 - 0.0004646980470803373j,
            -0.008884407577565362 + 0.005418552363686883j,
            0.2589463980023335 + 0.013353365347169917j,
        ]
        self.beam_grid_assertions(
            self.beam_avg_shape, ref_center_values, ref_freq_axis, mode, return_tuple
        )

    def test_grid_gaussian_beam_no_chan_average(self):
        mode = "gaussian no average"
        return_tuple = grid_beam(
            ant_ddi_xdt=self.ant_ddi_xdt,
            grid_size=self.grid_size,
            sky_cell_size=self.cell_size,
            avg_chan=False,
            telescope=self.vla,
            chan_tol_fac=self.chan_tolerance_factor,
            grid_interpolation_mode="gaussian",
            observation_summary=self.obs_sum,
            label=f"Test {mode}",
        )

        ref_freq_axis = self.ant_ddi_xdt[self.map_key].chan.values
        ref_center_values = [
            0.37580005477133005 + 0.019891983361750222j,
            0.013640750479123904 + 0.01227407036248688j,
            -0.011975895008132082 + 0.01375238260072988j,
            0.3630087962254655 + 0.015249794836502932j,
        ]
        self.beam_grid_assertions(
            self.beam_no_avg_shape,
            ref_center_values,
            ref_freq_axis,
            mode,
            return_tuple,
        )

    def test_grid_gaussian_beam_chan_average(self):
        mode = "gaussian average"
        return_tuple = grid_beam(
            ant_ddi_xdt=self.ant_ddi_xdt,
            grid_size=self.grid_size,
            sky_cell_size=self.cell_size,
            avg_chan=True,
            telescope=self.vla,
            chan_tol_fac=self.chan_tolerance_factor,
            grid_interpolation_mode="gaussian",
            observation_summary=self.obs_sum,
            label=f"Test {mode}",
        )

        ref_freq_axis = np.average(self.ant_ddi_xdt[self.map_key].chan.values)
        ref_center_values = [
            0.33047897570338663 + 0.018859501335040586j,
            0.005984131375211977 + 0.0023096517403009068j,
            -0.005063436214969807 + 0.010002728960915877j,
            0.323584178032157 + 0.017291480615071117j,
        ]
        self.beam_grid_assertions(
            self.beam_avg_shape,
            ref_center_values,
            ref_freq_axis,
            mode,
            return_tuple,
        )

    def test_gaussian_convolution_grid_correction(self):
        fake_aperture = np.full([1, 1, 1, 512, 512], 1.0 + 0j)
        fake_axis = simple_axis([-15, 15], 0.06458)
        vla = get_proper_telescope("vla")
        reference_lambda = 0.03
        freq = clight / reference_lambda
        cell_size = 0.85 * reference_lambda / vla.diameter
        sky_cell_size = np.array([-cell_size, cell_size])

        corr_aperture = gridding_correction(
            fake_aperture, freq, vla.diameter, sky_cell_size, fake_axis, fake_axis
        )

        reference_values = [
            [[256, 256], 1.0000487915970857 + 0j],
            [[310, 256], 1.0709547299089732 + 0j],
            [[128, 12], 6.490130669999345 + 0j],
            [[256, 140], 1.3965017152774735 + 0j],
            [[500, 256], 4.2229619313395865 + 0j],
        ]

        for idx, val in reference_values:
            assert np.isclose(corr_aperture[0, 0, 0, *idx], val), (
                f"Aperture correction at {idx} is not what was expected"
            )

    def test_1d_linear_gridding(self):
        method = "linear"
        self.resample_weight_test(method, True, 1.0)

        self.resample_weight_test(method, False, [1.0, 2.0, 3.0])

        self.resample_nan_count(method, False, 3)

        self.resample_nan_count(method, True, 0)

    def test_1d_gaussian_gridding(self):
        method = "gaussian"
        ref_weights = [0.00390625, 0.0078125, 1.0, 1.0]
        self.resample_weight_test(method, True, ref_weights)

        ref_weights = [
            1.25391006,
            1.50391006,
            1.50781631,
            1.50782013,
            1.50782013,
            1.50782013,
        ]
        self.resample_weight_test(method, False, ref_weights)
