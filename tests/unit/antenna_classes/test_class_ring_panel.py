import numpy as np

from astrohack.antenna.ring_panel import RingPanel


class TestRingPanel:
    in_radius = 2.0
    out_radius = 3.0
    angle = np.pi / 2
    i_panel = 1
    deviation = 2.0
    point = [2.5, -2.5, 1, 1, deviation]
    label = "test"
    margin = 0.2
    panel = RingPanel(
        "rigid", angle, i_panel, label, in_radius, out_radius, margin=margin
    )

    def test_init(self):
        """
        Tests the correct initialization of a RingPanel object, not all parameters tested
        """
        theta_margin = self.margin * self.angle
        radius_margin = self.margin * (self.out_radius - self.in_radius)
        zeta = (self.i_panel + 0.5) * self.angle
        rt = (self.in_radius + self.out_radius) / 2
        assert self.panel.theta1 == self.angle, "Panel initial angle is incorrect"
        assert self.panel.theta2 == 2 * self.angle, "Panel final angle is incorrect"
        assert self.panel.zeta == zeta, "Panel central angle is incorrect"
        assert self.panel.margin_theta1 == self.angle + theta_margin
        assert self.panel.margin_theta2 == 2 * self.angle - theta_margin
        assert self.panel.margin_inrad == self.in_radius + radius_margin
        assert self.panel.margin_ourad == self.out_radius - radius_margin
        assert self.panel.center.xc == -rt * np.sin(zeta)
        assert self.panel.center.yc == rt * np.cos(zeta)
        assert not self.panel.first

    def test_init_screws(self):
        """
        Test screw initialization
        """
        n_screws = 4
        scheme = None  # screws are at the corners of the panels
        offset = 0.0  # screws are precisely at the corners
        test_screws = np.zeros([n_screws, 2])
        test_screws[0, :] = [-np.sin(self.angle), np.cos(self.angle)]
        test_screws[1, :] = [-np.sin(2 * self.angle), np.cos(2 * self.angle)]
        test_screws[2, :] = [-np.sin(self.angle), np.cos(self.angle)]
        test_screws[3, :] = [-np.sin(2 * self.angle), np.cos(2 * self.angle)]

        test_screws[0:2, :] *= self.in_radius
        test_screws[2:, :] *= self.out_radius
        code_screws = self.panel._init_screws(scheme, offset)
        diff_sum = 0
        for i_screw, test_screw in enumerate(test_screws):
            diff_sum += code_screws[i_screw].xc - test_screw[0]
            diff_sum += code_screws[i_screw].yc - test_screw[1]
        assert code_screws.shape[0] == n_screws, (
            "If no scheme is given, there should be 4 screws at the corners"
        )
        assert np.abs(diff_sum) < 1e-15, "Screws with no offset do not match"

        offset = 6e-2  # 6 cm offset from panel edge
        radii = [
            self.in_radius + offset,
            self.in_radius + offset,
            self.out_radius - offset,
            self.out_radius - offset,
        ]
        theta = [
            self.angle + offset / radii[0],
            2 * self.angle - offset / radii[1],
            self.angle + offset / radii[2],
            2 * self.angle - offset / radii[3],
        ]
        for i in range(4):
            test_screws[i, :] = [
                -radii[i] * np.sin(theta[i]),
                radii[i] * np.cos(theta[i]),
            ]
        code_screws = self.panel._init_screws(scheme, offset)
        diff_sum = 0
        for i_screw, test_screw in enumerate(test_screws):
            diff_sum += code_screws[i_screw].xc - test_screw[0]
            diff_sum += code_screws[i_screw].yc - test_screw[1]

        assert np.abs(diff_sum) < 1e-15, "Screws with an offset do not match"
        scheme = ["c"]
        code_screws = self.panel._init_screws(scheme, offset)
        assert code_screws.shape[0] == 1, (
            "If scheme has a single screw, output must have a single screw"
        )
        diff_sum = (
            code_screws[0].xc
            - self.panel.center.xc
            + code_screws[0].yc
            - self.panel.center.yc
        )
        assert np.abs(diff_sum) < 1e-15, (
            "A center screw must be at the center of a panel"
        )
        return

    def test_is_inside(self):
        """
        Test over the is_inside test for a point
        """
        is_sample, is_in_panel = self.panel.is_inside(
            (self.in_radius + self.out_radius) / 2, 1.5 * self.angle
        )
        assert is_sample and is_in_panel, (
            "center of the panel must be a sample and inside panel"
        )
        is_sample, is_in_panel = self.panel.is_inside(
            (self.in_radius + self.out_radius) / 2, 3.5 * self.angle
        )
        assert (not is_sample) and (not is_in_panel), (
            "Point on the other side of the surface must be fully outside panel"
        )
        is_sample, is_in_panel = self.panel.is_inside(
            (self.in_radius + self.out_radius) / 2, 1.1 * self.angle
        )
        assert (not is_sample) and is_in_panel, (
            "Point at margin must be inside but not a sample"
        )
