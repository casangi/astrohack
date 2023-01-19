import pytest

from astrohack._classes.ring_panel import RingPanel, _gauss_elimination_numpy
import numpy as np

kinds = ['single', 'rigid', 'xyparaboloid', 'thetaparaboloid', 'fixedtheta']


class TestRingPanel:
    inrad = 2.0
    ourad = 3.0
    angle = np.pi/2
    position = 1
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
        point = [2.5, -2.5, 1, 1, 2.0]
        self.fixedthetapan.add_point(point)
        assert self.fixedthetapan.nsamp == 1
        assert self.fixedthetapan.values[0] == point
        assert len(self.fixedthetapan.values) == 1
