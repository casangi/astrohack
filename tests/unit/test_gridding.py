import numpy as np
import toolviper

from astrohack.utils import clight
from astrohack.utils.gridding import grid_beam, grid_1d_data, gridding_correction
from astrohack import get_proper_telescope, extract_holog, extract_pointing
from astrohack.utils.file import load_holog_file
from astrohack.utils.ray_tracing_general import simple_axis


class TestGridAlgorithms:
    chan_tolerance_factor = 0.005
    vla = get_proper_telescope('vla')
    datafolder = 'data'

    @classmethod
    def setup_class(cls):
        toolviper.utils.data.download(
            file="ea25_cal_small_before_fixed.split.ms", folder=cls.datafolder
        )

        extract_pointing(
            ms_name=cls.datafolder + "/ea25_cal_small_before_fixed.split.ms",
            point_name=cls.datafolder + "/ea25_cal_small_before_fixed.split.point.zarr",
            parallel=True,
            overwrite=True,
        )

        extract_holog(
            ms_name=cls.datafolder + "/ea25_cal_small_before_fixed.split.ms",
            point_name=cls.datafolder + "/ea25_cal_small_before_fixed.split.point.zarr",
            data_column="CORRECTED_DATA",
            ddi=0,
            parallel=True,
            overwrite=True,
        )

        cls.holog_mds, cls.ant_data_dict = load_holog_file(
            cls.datafolder + "/ea25_cal_small_before_fixed.split.holog.zarr",
            dask_load=False,
            load_pnt_dict=False,
            ant_id='ant_ea25',
            ddi_id='ddi_0',
        )

        cls.obs_sum = cls.ant_data_dict['ddi_0']['map_0'].attrs["summary"]
        cls.cell_size = np.array([-cls.obs_sum["beam"]["cell size"], cls.obs_sum["beam"]["cell size"]])
        cls.grid_size = np.array(cls.obs_sum["beam"]["grid size"])
        cls.expected_time_centroid = 5.16975892e+09

    def test_grid_linear_beam_no_chan_average(self):
        beam_grid, time_centroid, output_freq_axis, pol_axis, l_axis, m_axis, grid_corr, obs_sum = grid_beam(
            ant_ddi_dict=self.ant_data_dict['ddi_0'],
            grid_size=self.grid_size,
            sky_cell_size=self.cell_size,
            avg_chan=False,
            telescope=self.vla,
            chan_tol_fac=self.chan_tolerance_factor,
            grid_interpolation_mode='linear',
            observation_summary=self.obs_sum,
            label='Test beam linear grid',
        )

        ref_freq_axis = self.ant_data_dict['ddi_0']['map_0'].chan.values
        expected_shape = [1, 64, 4, 28, 28]


        ref_center_values = [0.2780216888736269+0.0029742640382882136j,
                             0.005223369026628943+0.018311068461403537j,
                             -0.007586815158936162+0.006465368831061344j,
                             0.2863594496065688+0.015925966825276847j
                             ]

        assert np.all(np.isclose(expected_shape, beam_grid.shape)), "Beam grid does not have the expected shape"

        for i_pol, pol in enumerate(pol_axis):
            assert np.isclose(beam_grid[0, 32, i_pol, 14, 14], ref_center_values[i_pol]), \
                f'Center pixel for {pol} does not match reference'

        assert np.isclose(time_centroid[0], self.expected_time_centroid), \
            "Time centroid is different from the expected"
        assert l_axis.shape[0] == self.grid_size[0], "Grid size and l axis size are not equal"
        assert m_axis.shape[0] == self.grid_size[1], "Grid size and m axis size are not equal"
        assert np.all(np.isclose(ref_freq_axis, output_freq_axis)), \
            "Output frequency axis is not equal to the reference"
        assert not grid_corr, "Linear beam griddind does not warrant a gridding correction in aperture plane"
        assert obs_sum == self.obs_sum, "Observation summary differs from the expected"

        return

    def test_grid_linear_beam_chan_average(self):
        beam_grid, time_centroid, output_freq_axis, pol_axis, l_axis, m_axis, grid_corr, obs_sum = grid_beam(
            ant_ddi_dict=self.ant_data_dict['ddi_0'],
            grid_size=self.grid_size,
            sky_cell_size=self.cell_size,
            avg_chan=True,
            telescope=self.vla,
            chan_tol_fac=self.chan_tolerance_factor,
            grid_interpolation_mode='linear',
            observation_summary=self.obs_sum,
            label='Test beam linear grid',
        )

        ref_freq_axis = np.average(self.ant_data_dict['ddi_0']['map_0'].chan.values)
        expected_shape = [1, 1, 4, 28, 28]


        ref_center_values = [0.2555271590565431+0.013577819885576325j,
                             0.006690232199874844-0.0004646980470803373j,
                             -0.008884407577565362+0.005418552363686883j,
                             0.2589463980023335+0.013353365347169917j,
                             ]

        assert np.all(np.isclose(expected_shape, beam_grid.shape)), "Beam grid does not have the expected shape"

        for i_pol, pol in enumerate(pol_axis):
            assert np.isclose(beam_grid[0, 0, i_pol, 14, 14], ref_center_values[i_pol]), \
                f'Center pixel for {pol} does not match reference'

        assert np.isclose(time_centroid[0], self.expected_time_centroid), \
            "Time centroid is different from the expected"
        assert l_axis.shape[0] == self.grid_size[0], "Grid size and l axis size are not equal"
        assert m_axis.shape[0] == self.grid_size[1], "Grid size and m axis size are not equal"
        assert np.all(np.isclose(ref_freq_axis, output_freq_axis)), \
            "Output frequency axis is not equal to the reference"
        assert not grid_corr, "Linear beam griddind does not warrant a gridding correction in aperture plane"
        assert obs_sum == self.obs_sum, "Observation summary differs from the expected"

        return

    def test_grid_gaussian_beam_no_chan_average(self):
        beam_grid, time_centroid, output_freq_axis, pol_axis, l_axis, m_axis, grid_corr, obs_sum = grid_beam(
            ant_ddi_dict=self.ant_data_dict['ddi_0'],
            grid_size=self.grid_size,
            sky_cell_size=self.cell_size,
            avg_chan=False,
            telescope=self.vla,
            chan_tol_fac=self.chan_tolerance_factor,
            grid_interpolation_mode='gaussian',
            observation_summary=self.obs_sum,
            label='Test beam linear grid',
        )

        ref_freq_axis = self.ant_data_dict['ddi_0']['map_0'].chan.values
        expected_shape = [1, 64, 4, 28, 28]


        ref_center_values = [0.37580005477133005+0.019891983361750222j,
                             0.013640750479123904+0.01227407036248688j,
                             -0.011975895008132082+0.01375238260072988j,
                             0.3630087962254655+0.015249794836502932j,
                             ]

        assert np.all(np.isclose(expected_shape, beam_grid.shape)), "Beam grid does not have the expected shape"

        for i_pol, pol in enumerate(pol_axis):
            assert np.isclose(beam_grid[0, 32, i_pol, 14, 14], ref_center_values[i_pol]), \
                f'Center pixel for {pol} does not match reference'

        assert np.isclose(time_centroid[0], self.expected_time_centroid), \
            "Time centroid is different from the expected"
        assert l_axis.shape[0] == self.grid_size[0], "Grid size and l axis size are not equal"
        assert m_axis.shape[0] == self.grid_size[1], "Grid size and m axis size are not equal"
        assert np.all(np.isclose(ref_freq_axis, output_freq_axis)), \
            "Output frequency axis is not equal to the reference"
        assert grid_corr, "Linear beam griddind does not warrant a gridding correction in aperture plane"
        assert obs_sum == self.obs_sum, "Observation summary differs from the expected"

        return

    def test_grid_gaussian_beam_chan_average(self):
        beam_grid, time_centroid, output_freq_axis, pol_axis, l_axis, m_axis, grid_corr, obs_sum = grid_beam(
            ant_ddi_dict=self.ant_data_dict['ddi_0'],
            grid_size=self.grid_size,
            sky_cell_size=self.cell_size,
            avg_chan=True,
            telescope=self.vla,
            chan_tol_fac=self.chan_tolerance_factor,
            grid_interpolation_mode='gaussian',
            observation_summary=self.obs_sum,
            label='Test beam linear grid',
        )

        ref_freq_axis = np.average(self.ant_data_dict['ddi_0']['map_0'].chan.values)
        expected_shape = [1, 1, 4, 28, 28]


        ref_center_values = [0.33047897570338663+0.018859501335040586j,
                             0.005984131375211977+0.0023096517403009068j,
                             -0.005063436214969807+0.010002728960915877j,
                             0.323584178032157+0.017291480615071117j,
                             ]

        assert np.all(np.isclose(expected_shape, beam_grid.shape)), "Beam grid does not have the expected shape"

        for i_pol, pol in enumerate(pol_axis):
            assert np.isclose(beam_grid[0, 0, i_pol, 14, 14], ref_center_values[i_pol]), \
                f'Center pixel for {pol} does not match reference'

        assert np.isclose(time_centroid[0], self.expected_time_centroid), \
            "Time centroid is different from the expected"
        assert l_axis.shape[0] == self.grid_size[0], "Grid size and l axis size are not equal"
        assert m_axis.shape[0] == self.grid_size[1], "Grid size and m axis size are not equal"
        assert np.all(np.isclose(ref_freq_axis, output_freq_axis)), \
            "Output frequency axis is not equal to the reference"
        assert grid_corr, "Linear beam griddind does not warrant a gridding correction in aperture plane"
        assert obs_sum == self.obs_sum, "Observation summary differs from the expected"

        return

    def test_gaussian_convolution_grid_correction(self):
        fake_aperture = np.full([1, 1, 1, 512, 512], 1.0 + 0j)
        fake_axis = simple_axis([-15, 15], 0.06458)
        vla = get_proper_telescope('vla')
        reference_lambda = 0.03
        freq = clight/reference_lambda
        cell_size = 0.85 * reference_lambda / vla.diameter
        sky_cell_size = np.array([-cell_size, cell_size])

        corr_aperture = gridding_correction(fake_aperture, freq, vla.diameter, sky_cell_size, fake_axis, fake_axis)

        reference_values = [
            [[256, 256], 1.0000487915970857+0j],
            [[310, 256], 1.0709547299089732+0j],
            [[128, 12], 6.490130669999345+0j],
            [[256, 140], 1.3965017152774735+0j],
            [[500, 256], 4.2229619313395865+0j],
        ]

        for idx, val in reference_values:
            assert np.isclose(corr_aperture[0,0,0, *idx], val), f"Aperture correction at {idx} is not what was expected"


