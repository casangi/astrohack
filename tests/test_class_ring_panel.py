import pytest
from astrohack._classes.ring_panel import RingPanel
import numpy as np


class TestRingPanel:
    inrad = 2.0
    ourad = 3.0
    angle = np.pi / 2
    position = 1
    deviation = 2.0
    point = [2.5, -2.5, 1, 1, deviation]
    corotatedpan = RingPanel('corotatedparaboloid', angle, position, position, inrad, ourad)

    def test_init(self):
        """
        Tests the correct initialization of a RingPanel object, not all parameters tested
        """
        assert self.corotatedpan.theta1 == self.angle, 'Panel initial angle is incorrect'
        assert self.corotatedpan.theta2 == 2 * self.angle, 'Panel final angle is incorrect'
        assert self.corotatedpan.zeta == 1.5 * self.angle, 'Panel central angle is incorrect'
        assert self.corotatedpan.iring == self.position+1, 'Panel ring numbering is incorrect'

    def test_is_inside(self):
        """
        Test over the is_inside test for a point
        """
        assert self.corotatedpan.is_inside((self.inrad + self.ourad) / 2, 1.5 * self.angle), 'Point that should be ' \
                                                                                             'inside the panel isn\'t'
        assert not self.corotatedpan.is_inside((self.inrad + self.ourad) / 2, 3.5 * self.angle), 'Point that should ' \
                                                                                                 'be outside the ' \
                                                                                                 'panel isn\'t'

    def test_export_adjustments(self):
        """
        Tests that panel adjustments are within what is expected from input data
        """
        npoints = 200
        for i in range(npoints):
            self.corotatedpan.add_point(self.point)
        self.corotatedpan.solve()
        self.corotatedpan.get_corrections()
        exportedstr = self.corotatedpan.export_adjustments().split()

        assert float(exportedstr[0]) == self.position + 1, 'Panel ring numbering is wrong when exported'
        assert float(exportedstr[1]) == self.position + 1, 'Panel numbering is wrong when exported'
