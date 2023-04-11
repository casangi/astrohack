import pytest
from astrohack._classes.ring_panel import RingPanel
import numpy as np


class TestRingPanel:
    inrad = 2.0
    ourad = 3.0
    angle = np.pi / 2
    ipanel = 1
    deviation = 2.0
    point = [2.5, -2.5, 1, 1, deviation]
    label = 'test'
    meta  = 'test_ant'
    margin = 0.2
    panel = RingPanel('rigid', angle, ipanel, label, inrad, ourad, meta, margin=margin)

    def test_init(self):
        """
        Tests the correct initialization of a RingPanel object, not all parameters tested
        """
        theta_margin = self.margin*self.angle
        radius_margin = self.margin*(self.ourad-self.inrad)
        assert self.panel.theta1 == self.angle, 'Panel initial angle is incorrect'
        assert self.panel.theta2 == 2 * self.angle, 'Panel final angle is incorrect'
        assert self.panel.zeta == 1.5 * self.angle, 'Panel central angle is incorrect'
        assert self.panel.margin_theta1 == self.angle + theta_margin
        assert self.panel.margin_theta2 == 2*self.angle - theta_margin
        assert self.panel.margin_inrad == self.inrad + radius_margin
        assert self.panel.margin_ourad == self.ourad - radius_margin
        #assert self.panel.center == 0.0
        assert not self.panel.first
        
        
    def test_is_inside(self):
        """
        Test over the is_inside test for a point
        """
        assert self.panel.is_inside((self.inrad + self.ourad) / 2, 1.5 * self.angle), 'Point that should be ' \
                                                                                             'inside the panel isn\'t'
        assert not self.panel.is_inside((self.inrad + self.ourad) / 2, 3.5 * self.angle), 'Point that should ' \
                                                                                                 'be outside the ' \
                                                                                                 'panel isn\'t'

    def test_export_adjustments(self):
        """
        Tests that panel adjustments are within what is expected from input data
        """
        npoints = 200
        for i in range(npoints):
            self.panel.add_point(self.point)
        self.panel.solve()
        self.panel.get_corrections()
        exportedstr = self.panel.export_adjustments().split()

        assert float(exportedstr[0]) == self.position + 1, 'Panel ring numbering is wrong when exported'
        assert float(exportedstr[1]) == self.position + 1, 'Panel numbering is wrong when exported'
