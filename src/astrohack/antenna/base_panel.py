import toolviper.utils.logger as logger

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from astrohack.antenna.panel_fitting import PANEL_MODEL_DICT, PanelPoint, PanelModel
from astrohack.utils.constants import *
from astrohack.utils import convert_unit

class BasePanel:
    markers = ['X', 'o', '*', 'P', 'D']
    colors = ['g', 'g', 'r', 'r', 'b']
    linewidth = 0.5
    linecolor = 'black'

    def __init__(self, model, screws, plot_screw_pos, plot_screw_size, label, center=None, zeta=None, ref_points=None):
        """
        Initializes the base panel with the appropriated fitting methods and providing basic functionality
        Fitting method models are:
        AIPS fitting models:
            mean: The panel is corrected by the mean of its samples
            rigid: The panel samples are fitted to a rigid surface
        Corotated Paraboloids (the two bending axes are parallel and perpendicular to the radius of the antenna crossing
        the middle of the panel):
            corotated_scipy: Paraboloid is fitted using scipy.optimize, robust but slow
            corotated_lst_sq: Paraboloid is fitted using the linear algebra least squares method, fast but unreliable
            corotated_robust: Tries corotated_lst_sq, if it diverges falls back to corotated_scipy
        Experimental fitting models:
            xy_paraboloid: fitted using scipy.optimize, bending axes are parallel to the x and y axes
            rotated_paraboloid: fitted using scipy.optimize, bending axes can be rotated by an arbitrary angle
            full_paraboloid_lst_sq: Full 9 parameter paraboloid fitted using least_squares method, heavily overfits
        Args:
            model: What model of surface fitting method to be used
            label: Panel label
            screws: position of the screws
            center: Panel center
            zeta: panel center angle
        """
        self.model_name = model
        self.solved = False
        self.fall_back_fit = False
        self.label = label
        self.screws = screws
        self.plot_screw_pos = plot_screw_pos
        self.plot_screw_size = plot_screw_size
        self.samples = []
        self.margins = []
        self.corr = None

        if center is None:
            self.center = PanelPoint(0, 0)
        else:
            self.center = center
        if zeta is None:
            self.zeta = 0
        else:
            self.zeta = zeta
        if ref_points is None:
            self.ref_points = [0, 0, 0]
        else:
            self.ref_points = ref_points

        try:
            model_dict = PANEL_MODEL_DICT[self.model_name]
        except KeyError:
            msg = f'Unknown model {self.model_name}'
            logger.error(msg)
            raise Exception(msg)

        self.model = PanelModel(model_dict, self.zeta, self.ref_points, self.center)

    def add_sample(self, sample):
        """
        Add a point to the panel's list of points to be fitted
        Args:
            sample: tuple/list containing point description [xcoor,ycoor,xidx,yidx,value]
        """
        if self.model_name in PANEL_MODEL_DICT.keys():
            self.samples.append(PanelPoint(*sample))
        else:
            self.samples.append(sample)

    def add_margin(self, sample):
        """
        Add a point to the panel's list of points to be corrected, but not fitted
        Args:
            sample: tuple/list containing point description [xcoor,ycoor,xidx,yidx,value]
        """
        if self.model_name in PANEL_MODEL_DICT.keys():
            self.margins.append(PanelPoint(*sample))
        else:
            self.margins.append(sample)

    def solve(self):
        if len(self.samples) < self.model.npar:
            self._fallback_solve()
            status = False
        else:
            try:
                self.model.solve(self.samples)
                status = True
            except np.linalg.LinAlgError:
                self._fallback_solve()
                status = False
        self.solved = True
        return status

    def _fallback_solve(self):
        """
        Changes the method association to mean surface fitting, and fits the panel with it
        """
        self.model = PanelModel(PANEL_MODEL_DICT['mean'], self.zeta, self.ref_points, self.center)
        self.model.solve(self.samples)
        self.fall_back_fit = True

    def get_corrections(self):
        if not self.solved:
            msg = 'Cannot correct a panel that is not solved'
            logger.error(msg)
            raise Exception(msg)
        self.corr = self.model.correct(self.samples, self.margins)
        return self.corr

    def export_screws(self, unit='mm'):
        """
        Export screw adjustments to a numpy array in unit
        Args:
            unit: Unit for the screw adjustments

        Returns:
            Numpy array with screw adjustments
        """
        fac = convert_unit('m', unit, 'length')
        nscrew = len(self.screws)
        screw_corr = np.zeros(nscrew)
        for iscrew, screw in enumerate(self.screws):
            screw_corr[iscrew] = fac * self.model.correct_point(screw)
        return screw_corr

    def plot_label(self, ax, rotate=True):
        """
        Plots panel label to ax
        Args:
            ax: matplotlib axes instance
            rotate: Rotate label for better display
        """
        if rotate:
            angle = (-self.zeta % pi - pi/2)*convert_unit('rad', 'deg', 'trigonometric')
        else:
            angle = 0
        ax.text(self.center.yc, self.center.xc, self.label, fontsize=fontsize, ha='center', va='center',
                rotation=angle)

    def plot_screws(self, ax):
        """
        Plots panel screws to ax
        Args:
            ax: matplotlib axes instance
        """
        for iscrew, screw in enumerate(self.screws):
            ax.scatter(screw.yc, screw.xc, marker=self.markers[iscrew], lw=self.linewidth, s=markersize,
                       color=self.colors[iscrew])

    def plot_corrections(self, ax, cmap, corrections, threshold, vmin, vmax):
        """
        Plot screw corrections onto an axis
        Args:
            ax: axis for plot
            cmap: Colormap of the corrections to be applied to each screw
            corrections: the screw corrections
            threshold: Threshold below which data is considered negligable
            vmin: bottom of the colormap
            vmax: top of the colormap
        """
        norm = Normalize(vmin=vmin, vmax=vmax)
        for iscrew in range(self.plot_screw_pos.shape[0]):
            screw = self.plot_screw_pos[iscrew]
            if np.abs(corrections[iscrew]) < threshold:
                corr = 0
            else:
                corr = corrections[iscrew]
            circle = plt.Circle((screw.yc, screw.xc), self.plot_screw_size, color=cmap(norm(corr)),
                                fill=True)
            ax.add_artist(circle)



