import pytest

import astrohack.antenna.panel_fitting as panfit
from astrohack.antenna.panel_fitting import PanelPoint
from astrohack.utils.algorithms import gauss_elimination
from astrohack.antenna.base_panel import BasePanel
from astrohack.utils.conversion import convert_unit
import numpy as np


class TestBasePanel:
    tolerance = 1e-6
    apoint = PanelPoint(0, 0)
    screws = [apoint, apoint, apoint, apoint]

    def testgauss_elimination(self):
        """
        Tests the gaussian elimination routine by using an identity matrix
        """
        size = 3
        identity = np.identity(size)
        vector = np.arange(size)
        for pos in range(size):
            assert gauss_elimination(identity, vector)[pos] == vector[pos], 'Gaussian elimination failed'

    def test_init(self):
        label = 'TEST'
        lepanel = BasePanel('mean', self.screws, False, 0.1, label)
        assert lepanel.label == label, "Internal panel label not what expected"
        assert lepanel.model_name == 'mean', "Internal model does not match input"
        assert lepanel.samples == [], 'List of samples should be empty'
        assert lepanel.margins == [], 'list of pixels in the margin should be empty'
        assert lepanel.corr is None, 'List of corrections should be None'
        assert not lepanel.solved, 'Panel cannot be solved at creation'
        with pytest.raises(Exception):
            lepanel = BasePanel('xxx', self.screws, False, 0.1, label)

    def test_add_point(self):
        """
        Test the add point common function
        """
        label = 'TEST'
        lepanel = BasePanel('mean', self.screws, False, 0.1, label)
        nsamp = 30
        point = [0, 0, 0, 0, 0]
        for i in range(nsamp):
            lepanel.add_sample(point)
            lepanel.add_margin(point)
        assert len(lepanel.samples) == nsamp, 'Internal number of samples do not match the expected number of samples'
        assert len(lepanel.margins) == nsamp, 'Internal list of points does not have the expected size'
        for i in range(nsamp):
            assert lepanel.samples[i].xc == point[0], '{0:d}-eth point does not match input point'.format(i)
        return

    def test_mean_model(self):
        """
        Tests the whole usage of a panel of the mean model
        """
        expectedmean = 3.5
        point = [0, 0, 0, 0, expectedmean]
        nsamp = 30
        meanpanel = BasePanel('mean', self.screws, False, 0.1, 'test')
        for i in range(nsamp):
            meanpanel.add_sample(point)
        meanpanel.solve()
        assert abs(meanpanel.model.parameters[0] - expectedmean)/expectedmean < self.tolerance, 'Did not recover the'\
                                                                                                ' expected mean'
        meanpanel.get_corrections()
        assert len(meanpanel.corr) == nsamp, 'Number of corrected points do not match number of samples'
        onecorr = meanpanel.model.correct_point(PanelPoint(0, 0))
        assert abs(onecorr - expectedmean)/expectedmean < self.tolerance, 'Correction for a point did not match the ' \
                                                                          'expected value'
        mmscrews = meanpanel.export_screws(unit='mm')
        fac = convert_unit('m', 'mm', 'length')
        for screw in mmscrews:
            assert abs(screw - fac*expectedmean) < self.tolerance, 'mm screw adjustments not within 0.1% tolerance '\
                                                                      'of the expected value'
        miscrews = meanpanel.export_screws(unit='mils')
        fac = convert_unit('m', 'mils', 'length')
        for screw in miscrews:
            assert abs(screw - fac * expectedmean) < 1e-2, 'Miliinches screw adjustments not ' \
                                                                         'within 1% of the expected value'

    def test_rigid_model(self):
        """
        Tests the whole usage of a panel of the rigid model
        """
        expectedpar = [3.5, -2, 1]
        nside = 32
        rigidpanel = BasePanel('rigid', self.screws, False, 0.1, 'test')

        for ix in range(nside):
            for iy in range(nside):
                value = ix*expectedpar[0] + iy*expectedpar[1] + expectedpar[2]
                rigidpanel.add_sample([ix, iy, ix, iy, value])
        rigidpanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert abs(rigidpanel.model.parameters[ipar]-expectedpar[ipar])/abs(expectedpar[ipar]) < self.tolerance, (
                feedback)
        rigidpanel.get_corrections()
        assert len(rigidpanel.corr) == nside**2, 'Number of corrected points do not match number of samples'
        onecorr = rigidpanel.model.correct_point(PanelPoint(0, 0))
        assert abs(onecorr - expectedpar[2])/expectedpar[2] < self.tolerance, 'Correction for a point did not match ' \
                                                                              'the expected value'
        mmscrews = rigidpanel.export_screws()
        fac = convert_unit('m', 'mm', 'length')
        for screw in mmscrews:
            assert abs(screw - fac*expectedpar[2]) < self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                     'tolerance of the expected value'

    def test_xyparaboloid_scipy_model(self):
        """
        Tests the whole usage of a panel of the xyparaboloid model
        """
        expectedpar = [150, 10, 2.5]
        nside = 32
        xyparapanel = BasePanel("xy_paraboloid", self.screws, False, 0.1, 'test')
        for ix in range(nside):
            for iy in range(nside):
                value = expectedpar[0]*ix**2 + expectedpar[1]*iy**2 + expectedpar[2]
                xyparapanel.add_sample([ix, iy, ix, iy, value])
        xyparapanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert (abs(xyparapanel.model.parameters[ipar] - expectedpar[ipar]) /
                    abs(expectedpar[ipar]) < self.tolerance), feedback
        xyparapanel.get_corrections()
        assert len(xyparapanel.corr) == nside ** 2, 'Number of corrected points do not match number of samples'
        onecorr = xyparapanel.model.correct_point(PanelPoint(0, 0))
        assert abs(onecorr - expectedpar[2]) / expectedpar[2] < self.tolerance, 'Correction for a point did not match '\
                                                                                'the expected value'
        mmscrews = xyparapanel.export_screws()
        fac = convert_unit('m', 'mm', 'length')
        for screw in mmscrews:
            assert abs(screw - fac*expectedpar[2]) < self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                        'tolerance of the expected value'

    def test_rotatedparaboloid_scipy_model(self):
        """
        Tests the whole usage of a panel of the rotatedparaboloid model
        """
        theta = 0
        expectedpar = [39, 10, 2.5, theta]
        nside = 32
        rotparapanel = BasePanel("rotated_paraboloid", self.screws, False, 0.1, 'test')
        for ix in range(nside):
            for iy in range(nside):
                u = ix * np.cos(theta) - iy * np.sin(theta)
                v = ix * np.sin(theta) + iy * np.cos(theta)
                value = expectedpar[0] * u**2 + expectedpar[1] * v**2 + expectedpar[2]
                rotparapanel.add_sample([ix, iy, ix, iy, value])
        rotparapanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert (abs(rotparapanel.model.parameters[ipar] - expectedpar[ipar]) /
                    abs(expectedpar[ipar]) < self.tolerance), feedback
        rotparapanel.get_corrections()
        assert len(rotparapanel.corr) == nside ** 2, 'Number of corrected points do not match number of samples'
        onecorr = rotparapanel.model.correct_point(PanelPoint(0, 0))
        assert abs(onecorr - expectedpar[2]) / expectedpar[2] < self.tolerance, 'Correction for a point did not match '\
                                                                                'the expected value'
        mmscrews = rotparapanel.export_screws()
        fac = convert_unit('m', 'mm', 'length')
        for screw in mmscrews:
            assert abs(screw - fac*expectedpar[2]) < 1e3*self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                        'tolerance of the expected value'

    def test_corotatedparaboloid_scipy_model(self):
        """
        Tests the whole usage of a panel of the corotatedparaboloid model solved with scipy
        """
        expectedpar = [75, 5, -2.0]
        nside = 32
        corotparapanel = BasePanel('corotated_scipy', self.screws, False, 0.1, 'test')
        for ix in range(nside):
            for iy in range(nside):
                value = expectedpar[0] * ix ** 2 + expectedpar[1] * iy ** 2 + expectedpar[2]
                corotparapanel.add_sample([ix, iy, ix, iy, value])
        corotparapanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert (abs(corotparapanel.model.parameters[ipar] - expectedpar[ipar]) /
                    abs(expectedpar[ipar]) < self.tolerance), feedback
        corotparapanel.get_corrections()
        assert len(corotparapanel.corr) == nside ** 2, 'Number of corrected points do not match number of samples'
        onecorr = corotparapanel.model.correct_point(PanelPoint(0, 0))
        assert abs(onecorr - expectedpar[2]) / expectedpar[2] < self.tolerance, 'Correction for a point did not match ' \
                                                                                'the expected value'
        mmscrews = corotparapanel.export_screws()
        fac = convert_unit('m', 'mm', 'length')
        for screw in mmscrews:
            assert abs(screw - fac * expectedpar[2]) < self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                       'tolerance of the expected value'

    def test_corotatedparaboloid_lst_model(self):
        """
        Tests the whole usage of a panel of the corotatedparaboloid model
        """
        expectedpar = [75, 5, -2.0]
        nside = 32
        corotparapanel = BasePanel("corotated_lst_sq", self.screws, False, 0.1, 'test')
        for ix in range(nside):
            for iy in range(nside):
                value = expectedpar[0] * ix ** 2 + expectedpar[1] * iy ** 2 + expectedpar[2]
                corotparapanel.add_sample([ix, iy, ix, iy, value])
        corotparapanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert (abs(corotparapanel.model.parameters[ipar] - expectedpar[ipar]) /
                    abs(expectedpar[ipar]) < self.tolerance), feedback
        corotparapanel.get_corrections()
        assert len(corotparapanel.corr) == nside ** 2, 'Number of corrected points do not match number of samples'
        onecorr = corotparapanel.model.correct_point(PanelPoint(0, 0))
        assert abs(onecorr - expectedpar[2]) / expectedpar[2] < self.tolerance, 'Correction for a point did not match ' \
                                                                                'the expected value'
        mmscrews = corotparapanel.export_screws()
        fac = convert_unit('m', 'mm', 'length')
        for screw in mmscrews:
            assert abs(screw - fac * expectedpar[2]) < self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                       'tolerance of the expected value'

    def test_corotatedparaboloid_robust_model(self):
        """
        Tests the whole usage of a panel of the corotatedparaboloid model
        """
        expectedpar = [75, 5, -2.0]
        nside = 32
        corotparapanel = BasePanel('corotated_robust', self.screws, False, 0.1, 'test')
        for ix in range(nside):
            for iy in range(nside):
                value = expectedpar[0] * ix ** 2 + expectedpar[1] * iy ** 2 + expectedpar[2]
                corotparapanel.add_sample([ix, iy, ix, iy, value])
        corotparapanel.solve()
        for ipar in range(3):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert (abs(corotparapanel.model.parameters[ipar] - expectedpar[ipar]) /
                    abs(expectedpar[ipar]) < self.tolerance), feedback
        corotparapanel.get_corrections()
        assert len(corotparapanel.corr) == nside ** 2, 'Number of corrected points do not match number of samples'
        onecorr = corotparapanel.model.correct_point(PanelPoint(0, 0))
        assert abs(onecorr - expectedpar[2]) / expectedpar[2] < self.tolerance, ('Correction for a point did not match '
                                                                                 'the expected value')
        mmscrews = corotparapanel.export_screws()
        fac = convert_unit('m', 'mm', 'length')
        for screw in mmscrews:
            assert abs(screw - fac * expectedpar[2]) < self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                       'tolerance of the expected value'

    def test_fullparaboloid_lst_model(self):
        """
        Tests the whole usage of a panel of the corotatedparaboloid model
        """
        expectedpar = [75, 5, -2.0, 3, -6, 8, 16, -3.5, 8]
        expectedscrew = 8000
        nside = 32
        fullpara = BasePanel("full_paraboloid_lst_sq", self.screws, False, 0.1, 'test')
        for ix in range(nside):
            for iy in range(nside):          
                xsq = ix**2
                ysq = iy**2
                value = expectedpar[0]*xsq*ysq + expectedpar[1]*xsq*iy + expectedpar[2]*ysq*ix
                value += expectedpar[3]*xsq + expectedpar[4]*ysq + expectedpar[5]*ix*iy
                value += expectedpar[6]*ix + expectedpar[7]*iy + expectedpar[8]
                fullpara.add_sample([ix, iy, ix, iy, value])
        fullpara.solve()
        for ipar in range(len(expectedpar)):
            feedback = '{0:d}-eth parameter does not match its expected value'.format(ipar)
            assert (abs(fullpara.model.parameters[ipar] - expectedpar[ipar]) /
                    abs(expectedpar[ipar]) < self.tolerance), feedback
        fullpara.get_corrections()
        assert len(fullpara.corr) == nside ** 2, 'Number of corrected points do not match number of samples'
        onecorr = fullpara.model.correct_point(PanelPoint(0, 0))
        assert abs(onecorr - expectedpar[2]) / expectedpar[2] < self.tolerance, ('Correction for a point did not match '
                                                                                 'the expected value')
        mmscrews = fullpara.export_screws()
        for screw in mmscrews:
            assert abs(screw - expectedscrew) < 1e3*self.tolerance, 'mm screw adjustments not within 0.1% ' \
                                                                          'tolerance of the expected value'
