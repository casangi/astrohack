from astrohack.antenna.ring_panel import RingPanel
import numpy as np


class TestRingPanel:
    inrad = 2.0
    ourad = 3.0
    angle = np.pi / 2
    ipanel = 1
    deviation = 2.0
    point = [2.5, -2.5, 1, 1, deviation]
    label = "test"
    margin = 0.2
    panel = RingPanel("rigid", angle, ipanel, label, inrad, ourad, margin=margin)

    def test_init(self):
        """
        Tests the correct initialization of a RingPanel object, not all parameters tested
        """
        theta_margin = self.margin * self.angle
        radius_margin = self.margin * (self.ourad - self.inrad)
        zeta = (self.ipanel + 0.5) * self.angle
        rt = (self.inrad + self.ourad) / 2
        assert self.panel.theta1 == self.angle, "Panel initial angle is incorrect"
        assert self.panel.theta2 == 2 * self.angle, "Panel final angle is incorrect"
        assert self.panel.zeta == zeta, "Panel central angle is incorrect"
        assert self.panel.margin_theta1 == self.angle + theta_margin
        assert self.panel.margin_theta2 == 2 * self.angle - theta_margin
        assert self.panel.margin_inrad == self.inrad + radius_margin
        assert self.panel.margin_ourad == self.ourad - radius_margin
        assert self.panel.center.xc == -rt * np.sin(zeta)
        assert self.panel.center.yc == rt * np.cos(zeta)
        assert not self.panel.first

    def test_init_screws(self):
        """
        Test screw initialization
        """
        nscrews = 4
        scheme = None  # screws are at the corners of the panels
        offset = 0.0  # screws are precisely at the corners
        test_screws = np.zeros([nscrews, 2])
        test_screws[0, :] = [-np.sin(self.angle), np.cos(self.angle)]
        test_screws[1, :] = [-np.sin(2 * self.angle), np.cos(2 * self.angle)]
        test_screws[2, :] = [-np.sin(self.angle), np.cos(self.angle)]
        test_screws[3, :] = [-np.sin(2 * self.angle), np.cos(2 * self.angle)]
        
        test_screws[0:2, :] *= self.inrad
        test_screws[2:, :] *= self.ourad
        code_screws = self.panel._init_screws(scheme, offset)
        diffsum = 0
        for iscrew, test_screw in enumerate(test_screws):
            diffsum += code_screws[iscrew].xc - test_screw[0]
            diffsum += code_screws[iscrew].yc - test_screw[1]
        assert (
            code_screws.shape[0] == nscrews
        ), "If no scheme is given, there should be 4 screws at the corners"
        assert np.abs(diffsum) < 1e-15, "Screws with no offset do not match"

        offset = 6e-2  # 6 cm offset from panel edge
        radii = [self.inrad + offset, self.inrad + offset, self.ourad - offset, self.ourad - offset]
        theta = [self.angle+offset/radii[0], 2*self.angle-offset/radii[1],
                 self.angle+offset/radii[2], 2*self.angle-offset/radii[3]]
        for i in range(4):
            test_screws[i, :] = [-radii[i]*np.sin(theta[i]), radii[i]*np.cos(theta[i])]
        code_screws = self.panel._init_screws(scheme, offset)
        diffsum = 0
        for iscrew, test_screw in enumerate(test_screws):
            diffsum += code_screws[iscrew].xc - test_screw[0]
            diffsum += code_screws[iscrew].yc - test_screw[1]

        assert np.abs(diffsum) < 1e-15, "Screws with an offset do not match"
        scheme = ["c"]
        code_screws = self.panel._init_screws(scheme, offset)
        assert (
            code_screws.shape[0] == 1
        ), "If scheme has a single screw, output must have a single screw"
        diffsum = (
            code_screws[0].xc
            - self.panel.center.xc
            + code_screws[0].yc
            - self.panel.center.yc
        )
        assert (
            np.abs(diffsum) < 1e-15
        ), "A center screw must be at the center of a panel"
        return

    def test_is_inside(self):
        """
        Test over the is_inside test for a point
        """
        issample, isinpanel = self.panel.is_inside(
            (self.inrad + self.ourad) / 2, 1.5 * self.angle
        )
        assert (
            issample and isinpanel
        ), "center of the panel must be a sample and inside panel"
        issample, isinpanel = self.panel.is_inside(
            (self.inrad + self.ourad) / 2, 3.5 * self.angle
        )
        assert (not issample) and (
            not isinpanel
        ), "Point on the other side of the surface must be fully outside panel"
        issample, isinpanel = self.panel.is_inside(
            (self.inrad + self.ourad) / 2, 1.1 * self.angle
        )
        assert (
            not issample
        ) and isinpanel, "Point at margin must be inside but not a sample"
