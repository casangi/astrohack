import pytest

from astrohack._classes.ring_panel import RingPanel, _gauss_elimination_numpy
import numpy as np

kinds = ['single', 'rigid', 'xyparaboloid', 'thetaparaboloid', 'fixedtheta']
mm2mi = 1000 / 25.4


class TestRingPanel:
    inrad = 2.0
    ourad = 3.0
    angle = np.pi/2
    position = 1
    deviation = 2.0
    point = [2.5, -2.5, 1, 1, deviation]
    fixedthetapan = RingPanel('fixedtheta', angle, position, position, inrad, ourad)

    def test_gauss_elimination_numpy(self):
        size = 3
        identity = np.identity(size)
        vector = np.arange(size)
        for pos in range(size):
            assert _gauss_elimination_numpy(identity, vector)[pos] == vector[pos]

    def test_init(self):
        assert self.fixedthetapan.theta1 == self.angle
        assert self.fixedthetapan.theta2 == 2 * self.angle
        assert self.fixedthetapan.zeta == 1.5 * self.angle

        with pytest.raises(Exception):
            panel = RingPanel('xxx', self.angle, self.position, self.position, self.inrad, self.ourad)

    def test_is_inside(self):
        assert self.fixedthetapan.is_inside((self.inrad + self.ourad) / 2, 1.5 * self.angle)
        assert not self.fixedthetapan.is_inside((self.inrad + self.ourad) / 2, 3.5 * self.angle)

    def test_add_point(self):
        self.fixedthetapan.add_point(self.point)
        assert self.fixedthetapan.nsamp == 1
        assert self.fixedthetapan.values[0] == self.point
        assert len(self.fixedthetapan.values) == 1

    def test_solve(self):
        npoints = 200
        for i in range(npoints):
            self.fixedthetapan.add_point(self.point)
        self.fixedthetapan.solve()
        assert self.fixedthetapan.solved
        assert self.fixedthetapan.par[0] > 1e2
        assert self.fixedthetapan.par[1] > 1e2
        assert abs(self.fixedthetapan.par[2] - self.deviation) < 1e-3

    def test_get_correction(self):
        self.fixedthetapan.get_corrections()
        assert len(self.fixedthetapan.corr) == self.fixedthetapan.nsamp
        for isamp in range(self.fixedthetapan.nsamp):
            assert abs(self.fixedthetapan.corr[isamp] - self.deviation) < 1e-3

    def test_export_adjustments(self):
        mmscrews = self.fixedthetapan.export_adjustments().split()[2:]
        for screw in mmscrews:
            assert abs(float(screw) - self.deviation) < 1e-3
        miscrews = self.fixedthetapan.export_adjustments(unit='miliinches').split()[2:]
        for screw in miscrews:
            print(screw, mm2mi*self.deviation)
            assert abs(float(screw) - mm2mi*self.deviation) < 1e-2
