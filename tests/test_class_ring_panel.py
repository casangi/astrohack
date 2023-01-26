import pytest
from astrohack._classes.ring_panel import RingPanel
import numpy as np

mm2mi = 1000 / 25.4


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
        assert self.corotatedpan.solve == self.corotatedpan._solve_scipy, 'solve method overloading failed'
        assert self.corotatedpan.corr_point == self.corotatedpan._corr_point_scipy, 'corr_point method overloading ' \
                                                                                    'failed'
        assert self.corotatedpan._paraboloid == self.corotatedpan._corotated_paraboloid, 'Paraboloid method ' \
                                                                                         'overload failed'

        with pytest.raises(Exception):
            panel = RingPanel(
                "xxx", self.angle, self.position, self.position, self.inrad, self.ourad
            )

    def test_is_inside(self):
        """
        Test over the is_inside test for a point
        """
        assert self.corotatedpan.is_inside((self.inrad + self.ourad) / 2, 1.5 * self.angle), 'Point that should be ' \
                                                                                             'inside the panel isn\'t'
        assert not self.corotatedpan.is_inside((self.inrad + self.ourad) / 2, 3.5 * self.angle), 'Point that should ' \
                                                                                                 'be outside the ' \
                                                                                                 'panel isn\'t'

    def test_add_point(self):
        """
        Tests the addition of a point to the panel list of points
        """
        self.corotatedpan.add_point(self.point)
        assert self.corotatedpan.nsamp == 1, 'Failed to increase number of samples after adding a point'
        assert self.corotatedpan.values[0] == self.point, 'Added point do not correspond to the data added'
        assert len(self.corotatedpan.values) == 1, 'Lenght of values list does not match its size counter'

    def test_solve(self):
        """
        Tests the solving of the panel surface but using only the scipy fitting of the fixedtheta version
        """
        npoints = 200
        for i in range(npoints):
            self.corotatedpan.add_point(self.point)
        self.corotatedpan.solve()
        assert self.corotatedpan.solved, 'Panel solving failed'
        assert self.corotatedpan.par[0] > 1e2, 'Panel curvature is smaller than expected'
        assert self.corotatedpan.par[1] > 1e2, 'Panel curvature is smaller than expected'
        assert abs(self.corotatedpan.par[2] - self.deviation) < 1e-3, 'Panel Z offset not within 0.1% tolerance'

    def test_get_correction(self):
        """
        Tests that corrections are what are expected based on the input data
        """
        self.corotatedpan.get_corrections()
        assert len(self.corotatedpan.corr) == self.corotatedpan.nsamp, 'Length of corrections array is not the same' \
                                                                       ' as the number of samples'
        for isamp in range(self.corotatedpan.nsamp):
            assert abs(self.corotatedpan.corr[isamp] - self.deviation) < 1e-3, 'Corrections not within 0.1% tolerance'

    def test_export_adjustments(self):
        """
        Tests that panel adjustments are within what is expected from input data
        """
        mmscrews = self.corotatedpan.export_adjustments().split()[2:]
        for screw in mmscrews:
            assert abs(float(screw) - self.deviation) < 1e-3, 'mm screw adjustments not within 0.1% tolerance of the ' \
                                                              'expected value'
        miscrews = self.corotatedpan.export_adjustments(unit='miliinches').split()[2:]
        for screw in miscrews:
            assert abs(float(screw) - mm2mi * self.deviation) < 1e-2, 'Miliinches screw adjustments not within 0.1% ' \
                                                                      'tolerance of the expected value'
