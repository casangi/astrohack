import numpy as np
import pytest

from astrohack.antenna.base_panel import BasePanel
from astrohack.antenna.panel_fitting import PanelPoint
from astrohack.utils.algorithms import gauss_elimination
from astrohack.utils.conversion import convert_unit


def rigid_value(ix, iy, expected_par):
    return ix * expected_par[0] + iy * expected_par[1] + expected_par[2]


def mean_value(ix, iy, expected_pars):
    return expected_pars[0]


def xy_paraboloid_value(ix, iy, expected_pars):
    return expected_pars[0] * ix**2 + expected_pars[1] * iy**2 + expected_pars[2]


def rotated_paraboloid_value(ix, iy, expected_pars):
    theta = expected_pars[3]
    u_coord = ix * np.cos(theta) - iy * np.sin(theta)
    v_coord = ix * np.sin(theta) + iy * np.cos(theta)
    value = (
        expected_pars[0] * u_coord**2 + expected_pars[1] * v_coord**2 + expected_pars[2]
    )
    return value


def corotated_paraboloid_value(ix, iy, expected_pars):
    return expected_pars[0] * ix**2 + expected_pars[1] * iy**2 + expected_pars[2]


def full_paraboloid_value(ix, iy, expected_pars):
    xsq = ix**2
    ysq = iy**2
    value = (
        expected_pars[0] * xsq * ysq
        + expected_pars[1] * xsq * iy
        + expected_pars[2] * ysq * ix
    )
    value += (
        expected_pars[3] * xsq + expected_pars[4] * ysq + expected_pars[5] * ix * iy
    )
    value += expected_pars[6] * ix + expected_pars[7] * iy + expected_pars[8]
    return value


class TestBasePanel:
    tolerance = 1e-6
    a_point = PanelPoint(0, 0)
    screws = [a_point, a_point, a_point, a_point]

    def panel_solving_assertions(
        self,
        panel_model,
        expected_pars,
        value_generator,
        n_side=32,
    ):
        this_panel = BasePanel(
            panel_model, self.screws, False, 0.1, f"{panel_model} test"
        )

        for ix in range(n_side):
            for iy in range(n_side):
                value = value_generator(ix, iy, expected_pars)
                this_panel.add_sample([ix, iy, ix, iy, value])

        n_par = len(expected_pars)
        this_panel.solve()
        for i_par in range(n_par):
            assert np.allclose(
                this_panel.model.parameters[i_par],
                expected_pars[i_par],
                atol=self.tolerance,
            ), f"{i_par}-eth parameter does not match its expected value"

        this_panel.get_corrections()
        assert len(this_panel.corr) == n_side**2, (
            "Number of corrected points do not match number of samples"
        )

        one_corr = this_panel.model.correct_point(PanelPoint(0, 0))
        assert np.isclose(
            one_corr, value_generator(0, 0, expected_pars), atol=self.tolerance
        ), "Correction for a point did not match the expected value"

        mm_screws = this_panel.export_screws()
        fac = convert_unit("m", "mm", "length")
        for screw_corr in mm_screws:
            assert np.isclose(
                screw_corr,
                fac * value_generator(0, 0, expected_pars),
                atol=self.tolerance,
            ), "mm screw adjustments not within 0.1% tolerance of the expected value"

    def test_gauss_elimination(self):
        """
        Tests the gaussian elimination routine by using an identity matrix
        """
        size = 3
        identity = np.identity(size)
        vector = np.arange(size)
        for pos in range(size):
            assert gauss_elimination(identity, vector)[pos] == vector[pos], (
                "Gaussian elimination failed"
            )

    def test_init(self):
        label = "TEST"
        this_panel = BasePanel("mean", self.screws, False, 0.1, label)
        assert this_panel.label == label, "Internal panel label not what expected"
        assert this_panel.model_name == "mean", "Internal model does not match input"
        assert this_panel.samples == [], "List of samples should be empty"
        assert this_panel.margins == [], "list of pixels in the margin should be empty"
        assert this_panel.corr is None, "List of corrections should be None"
        assert not this_panel.solved, "Panel cannot be solved at creation"
        with pytest.raises(ValueError):
            BasePanel("xxx", self.screws, False, 0.1, label)

    def test_add_point(self):
        """
        Test the add point common function
        """
        label = "TEST"
        this_panel = BasePanel("mean", self.screws, False, 0.1, label)
        n_samp = 30
        point = [0, 0, 0, 0, 0]
        for i in range(n_samp):
            this_panel.add_sample(point)
            this_panel.add_margin(point)
        assert len(this_panel.samples) == n_samp, (
            "Internal number of samples do not match the expected number of samples"
        )
        assert len(this_panel.margins) == n_samp, (
            "Internal list of points does not have the expected size"
        )
        for i in range(n_samp):
            assert this_panel.samples[i].xc == point[0], (
                f"{i:d}-eth point does not match input point"
            )
        return

    def test_fitting(self):
        fit_dict = {
            "mean": [[3.5], mean_value],
            "rigid": [[3.5, -2, 1], rigid_value],
            "xy_paraboloid": [[150, 10, 2.5], xy_paraboloid_value],
            "rotated_paraboloid": [[39, 10, 2.5, 0.0], rotated_paraboloid_value],
            "corotated_scipy": [[75, 5, -2.0], corotated_paraboloid_value],
            "corotated_lst_sq": [[75, 5, -2.0], corotated_paraboloid_value],
            "corotated_robust": [[75, 5, -2.0], corotated_paraboloid_value],
            "full_paraboloid_lst_sq": [
                [75, 5, -2.0, 3, -6, 8, 16, -3.5, 8],
                full_paraboloid_value,
            ],
        }

        for key, par_tuple in fit_dict.items():
            expected_pars, value_generator = par_tuple
            self.panel_solving_assertions(key, expected_pars, value_generator)
