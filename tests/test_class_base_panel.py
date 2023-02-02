import pytest

from astrohack._classes.base_panel import _gauss_elimination_numpy, BasePanel, \
     panelkinds, imean, irigid, ixypara, icorpara, irotpara
import numpy as np


class TestBasePanel:
    tolerance = 1e-6
    mm2mi = 1000 / 25.4

    def test_gauss_elimination_numpy(self):
        """
        Tests the gaussian elimination routine by using an identity matrix
        """
        size = 3
        identity = np.identity(size)
        vector = np.arange(size)
        for pos in range(size):
            assert _gauss_elimination_numpy(identity, vector)[pos] == vector[pos], 'Gaussian elimination failed'

    def test_init(self):
        screws = np.zeros([4, 2])
        ipanel = 0
        lepanel = BasePanel(panelkinds[imean], ipanel, screws)
        assert lepanel.ipanel == ipanel+1, "Internal ipanel number not what expected"
        assert lepanel.kind == panelkinds[imean], "Internal kind does not match input"
        assert lepanel.nsamp == 0, 'Number of samples should be 0'
        assert lepanel.values == [], 'List of values should be empty'
        assert lepanel.corr is None, 'List of corrections should be None'
        assert not lepanel.solved, 'Panel cannot be solved at creation'
        with pytest.raises(Exception):
            lepanel = BasePanel('xxx', ipanel, screws)

    def test_add_point(self):
        """
        Test the add point common function
        """
        screws = np.zeros([4, 2])
        ipanel = 0
        lepanel = BasePanel(panelkinds[imean], ipanel, screws)
        nsamp = 30
        point = [0, 0, 0, 0, 0]
        for i in range(nsamp):
            lepanel.add_point(point)
        assert lepanel.nsamp == nsamp, 'Internal number of samples do not match the expected number of samples'
        assert len(lepanel.values) == nsamp, 'Internal list of points does not have the expected size'
        for i in range(nsamp):
            assert lepanel.values[i] == point, '{0:d}-eth point does not match input point'.format(i)
        return

    def test_mean_kind(self):
        """
        Tests the whole usage of a panel of the mean kind
        """
        expectedmean = 3.5
        point = [0, 0, 0, 0, expectedmean]
        screws = np.zeros([4, 2])
        nsamp = 30
        ipanel = 0
        meanpanel = BasePanel(panelkinds[imean], ipanel, screws)
        assert meanpanel.solve == meanpanel._solve_mean, 'Incorrect overloading of mean solving method'
        assert meanpanel.corr_point == meanpanel._corr_point_mean, 'Incorrect overloading of mean point correction ' \
                                                                   'method'
        for i in range(nsamp):
            meanpanel.add_point(point)
        meanpanel.solve()
        assert abs(meanpanel.par[0] - expectedmean)/expectedmean < self.tolerance, 'Did not recover the expected mean'
        meanpanel.get_corrections()
        assert len(meanpanel.corr) == nsamp, 'Number of corrected points do not match number of samples'
        onecorr = meanpanel.corr_point(0, 0)
        assert abs(onecorr - expectedmean)/expectedmean < self.tolerance, 'Correction for a point did not match the ' \
                                                                          'expected value'
        mmscrews = meanpanel.export_screw_adjustments().split()
        for screw in mmscrews:
            assert abs(float(screw) - expectedmean) < self.tolerance, 'mm screw adjustments not within 0.1% tolerance '\
                                                                      'of the expected value'
        miscrews = meanpanel.export_screw_adjustments(unit='miliinches').split()
        for screw in miscrews:
            assert abs(float(screw) - self.mm2mi * expectedmean) < 1e-2, 'Miliinches screw adjustments not ' \
                                                                         'within 1% of the expected value'

    def test_rigid_kind(self):
        """
        Tests the whole usage of a panel of the rigid kind
        """
        expectedpar = [3.5, -2, 1]
        screws = np.zeros([4, 2])
        nside = 32
        ipanel = 0
        rigidpanel = BasePanel(panelkinds[irigid], ipanel, screws)
        assert rigidpanel.solve == rigidpanel._solve_rigid, 'Incorrect overloading of rigid solving method'
        assert rigidpanel.corr_point == rigidpanel._corr_point_rigid, 'Incorrect overloading of rigid point ' \
                                                                      'correction method'
        for ix in range(nside):
            for iy in range(nside):
                value = ix*expectedpar[0] + iy*expectedpar[1] + expectedpar[2]
                rigidpanel.add_point([ix, iy, ix, iy, value])
        rigidpanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert abs(rigidpanel.par[ipar]-expectedpar[ipar])/abs(expectedpar[ipar]) < self.tolerance, feedback
        rigidpanel.get_corrections()
        assert len(rigidpanel.corr) == nside**2, 'Number of corrected points do not match number of samples'
        onecorr = rigidpanel.corr_point(0, 0)
        assert abs(onecorr - expectedpar[2])/expectedpar[2] < self.tolerance, 'Correction for a point did not match ' \
                                                                              'the expected value'
        mmscrews = rigidpanel.export_screw_adjustments().split()
        for screw in mmscrews:
            assert abs(float(screw) - expectedpar[2]) < self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                        'tolerance of the expected value'

    def test_xyparaboloid_kind(self):
        """
        Tests the whole usage of a panel of the xyparaboloid kind
        """
        expectedpar = [150, 10, 2.5]
        screws = np.zeros([4, 2])
        nside = 32
        ipanel = 0
        xyparapanel = BasePanel(panelkinds[ixypara], ipanel, screws)
        assert xyparapanel.solve == xyparapanel._solve_scipy, 'Incorrect overloading of rigid solving method'
        assert xyparapanel.corr_point == xyparapanel._corr_point_scipy, 'Incorrect overloading of rigid point ' \
                                                                        'correction method'
        assert xyparapanel._paraboloid == xyparapanel._xyaxes_paraboloid, 'Incorrect overloading of paraboloid function'
        for ix in range(nside):
            for iy in range(nside):
                value = -((ix / expectedpar[0]) ** 2 + (iy / expectedpar[1]) ** 2) + expectedpar[2]
                xyparapanel.add_point([ix, iy, ix, iy, value])
        xyparapanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert abs(xyparapanel.par[ipar] - expectedpar[ipar]) / abs(expectedpar[ipar]) < self.tolerance, feedback
        xyparapanel.get_corrections()
        assert len(xyparapanel.corr) == nside ** 2, 'Number of corrected points do not match number of samples'
        onecorr = xyparapanel.corr_point(0, 0)
        assert abs(onecorr - expectedpar[2]) / expectedpar[2] < self.tolerance, 'Correction for a point did not match '\
                                                                                'the expected value'
        mmscrews = xyparapanel.export_screw_adjustments().split()
        for screw in mmscrews:
            assert abs(float(screw) - expectedpar[2]) < self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                        'tolerance of the expected value'

    def test_rotatedparaboloid_kind(self):
        """
        Tests the whole usage of a panel of the rotatedparaboloid kind
        """
        expectedpar = [39, 10, 2.5, np.pi/2]
        screws = np.zeros([4, 2])
        nside = 32
        ipanel = 0
        rotparapanel = BasePanel(panelkinds[irotpara], ipanel, screws)
        assert rotparapanel.solve == rotparapanel._solve_scipy, 'Incorrect overloading of rigid solving method'
        assert rotparapanel.corr_point == rotparapanel._corr_point_scipy, 'Incorrect overloading of rigid point ' \
                                                                          'correction method'
        assert rotparapanel._paraboloid == rotparapanel._rotated_paraboloid, 'Incorrect overloading of paraboloid ' \
                                                                             'function'
        for ix in range(nside):
            for iy in range(nside):
                value = -((iy / expectedpar[0]) ** 2 + (ix / expectedpar[1]) ** 2) + expectedpar[2]
                rotparapanel.add_point([ix, iy, ix, iy, value])
        rotparapanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert abs(rotparapanel.par[ipar] - expectedpar[ipar]) / abs(expectedpar[ipar]) < self.tolerance, feedback
        rotparapanel.get_corrections()
        assert len(rotparapanel.corr) == nside ** 2, 'Number of corrected points do not match number of samples'
        onecorr = rotparapanel.corr_point(0, 0)
        assert abs(onecorr - expectedpar[2]) / expectedpar[2] < self.tolerance, 'Correction for a point did not match '\
                                                                                'the expected value'
        mmscrews = rotparapanel.export_screw_adjustments().split()
        for screw in mmscrews:
            assert abs(float(screw) - expectedpar[2]) < self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                        'tolerance of the expected value'

    def test_corotatedparaboloid_kind(self):
        """
        Tests the whole usage of a panel of the corotatedparaboloid kind
        """
        expectedpar = [75, 5, -2.0]
        screws = np.zeros([4, 2])
        nside = 32
        ipanel = 0
        corotparapanel = BasePanel(panelkinds[icorpara], ipanel, screws)
        corotparapanel.zeta = np.pi/2
        assert corotparapanel.solve == corotparapanel._solve_scipy, 'Incorrect overloading of rigid solving method'
        assert corotparapanel.corr_point == corotparapanel._corr_point_scipy, 'Incorrect overloading of rigid point ' \
                                                                              'correction method'
        assert corotparapanel._paraboloid == corotparapanel._corotated_paraboloid, 'Incorrect overloading of ' \
                                                                                   'paraboloid function'
        for ix in range(nside):
            for iy in range(nside):
                value = -((iy / expectedpar[0]) ** 2 + (ix / expectedpar[1]) ** 2) + expectedpar[2]
                corotparapanel.add_point([ix, iy, ix, iy, value])
        corotparapanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert abs(corotparapanel.par[ipar] - expectedpar[ipar]) / abs(expectedpar[ipar]) < self.tolerance, feedback
        corotparapanel.get_corrections()
        assert len(corotparapanel.corr) == nside ** 2, 'Number of corrected points do not match number of samples'
        onecorr = corotparapanel.corr_point(0, 0)
        assert abs(onecorr - expectedpar[2]) / expectedpar[2] < self.tolerance, 'Correction for a point did not match '\
                                                                                'the expected value'
        mmscrews = corotparapanel.export_screw_adjustments().split()
        for screw in mmscrews:
            assert abs(float(screw) - expectedpar[2]) < self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                        'tolerance of the expected value'
