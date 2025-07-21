import numpy as np
import toolviper
from astrohack.utils.gridding import grid_beam, grid_1d_data
from astrohack import get_proper_telescope, extract_holog, extract_pointing
from astrohack.utils.file import load_holog_file


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

