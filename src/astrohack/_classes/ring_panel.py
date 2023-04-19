import numpy as np
from matplotlib import pyplot as plt
from astrohack._classes.base_panel import BasePanel
from astrohack._utils._constants import twopi


class RingPanel(BasePanel):
    # This class describes and treats panels that are arranged in
    # rings on the Antenna surface

    def __init__(self, kind, angle, ipanel, label, inrad, ourad, margin=0.20, screw_scheme=None, screw_offset=None):
        """
        Initializes a panel that is a section of a ring in a circular antenna
        Fitting method kinds are:
        AIPS fitting kinds:
            mean: The panel is corrected by the mean of its samples
            rigid: The panel samples are fitted to a rigid surface
        Corotated Paraboloids (the two bending axes are parallel and perpendicular to the radius of the antenna crossing
        the middle of the panel):
            corotated_scipy: Paraboloid is fitted using scipy.optimize, robust but slow
            corotated_lst_sq: Paraboloid is fitted using the linear algebra least squares method, fast but unreliable
            corotated_robust: Tries corotated_lst_sq, if it diverges falls back to corotated_scipy
        Experimental fitting kinds:
            xy_paraboloid: fitted using scipy.optimize, bending axes are parallel to the x and y axes
            rotated_paraboloid: fitted using scipy.optimize, bending axes can be rotated by an arbitrary angle
            full_paraboloid_lst_sq: Full 9 parameter paraboloid fitted using least_squares method, heavily overfits
        Args:
            kind: What kind of surface fitting method to be used
            angle: Azimuthal span of the panel
            ipanel: Panel number clockwise from top
            label: Panel label
            inrad: Radius at panel inner side
            ourad: Radius at panel outer side
            margin: Fraction from panel edge inwards that won't be used for fitting
            screw_scheme: tuple containing the description of screw positions
        """
        self.inrad = inrad
        self.ourad = ourad
        self.theta1 = ipanel * angle
        self.theta2 = (ipanel + 1) * angle
        dradius = ourad - inrad
        self.margin_theta1 = self.theta1 + margin * angle
        self.margin_theta2 = self.theta2 - margin * angle
        self.margin_inrad = inrad + margin * dradius
        self.margin_ourad = ourad - margin * dradius
        self.first = ipanel == 0
        zeta = (ipanel + 0.5) * angle
        rt = (self.inrad + self.ourad) / 2
        self.center = [rt * np.cos(zeta), rt * np.sin(zeta)]
        screws = self._init_screws(screw_scheme, screw_offset)
        # Now we are ready to initialize the base object
        super().__init__(kind, screws, label, center=self.center, zeta=zeta)

    def _init_screws(self, scheme, offset):
        """
        Initialize screws according to the scheme
        Args:
            scheme: Tuple of strings containing the positioning of the screws
            offset: How far from the edge of the panel are corner screws (meters)

        Returns:
            numpy array with the positions of the screws
        """
        if scheme is None:
            scheme = ['il', 'ir', 'ol', 'or']
        if offset is None:
            offset = 1e-2  # 1 cm
        nscrews = len(scheme)
        screws = np.ndarray([nscrews, 2])

        for iscrew in range(nscrews):
            if scheme[iscrew] == 'c':
                screws[iscrew, :] = self.center
            else:
                if scheme[iscrew][0] == 'i':
                    radius = self.inrad + offset
                else:
                    radius = self.ourad - offset
                deltatheta = offset / radius
                if scheme[iscrew][1] == 'l':
                    theta = self.theta1 + deltatheta
                else:
                    theta = self.theta2 - deltatheta
                screws[iscrew] = radius*np.cos(theta), radius*np.sin(theta)
        return screws

    def is_inside(self, rad, phi):
        """
        Check if a point is inside a panel using polar coordinates
        Args:
            rad: radius of the point
            phi: angle of the point in polar coordinates

        Returns:
            issample: True if point is inside the fitting part of the panel
            inpanel: True if point is inside the panel
        """
        # Simple test of polar coordinates to check that a point is
        # inside this panel
        angle = self.theta1 <= phi <= self.theta2
        radius = self.inrad <= rad <= self.ourad
        inpanel = angle and radius
        angle = self.margin_theta1 <= phi <= self.margin_theta2
        radius = self.margin_inrad <= rad <= self.margin_ourad
        issample = angle and radius
        return issample, inpanel

    def print_misc(self):
        """
        Print miscelaneous information about the panel to the terminal
        """
        print("########################################")
        print("{0:20s}={1:8s}".format("ipanel", self.label))
        print("{0:20s}={1:8s}".format("kind", " " + self.kind))
        print("{0:20s}={1:8.5f}".format("inrad", self.inrad))
        print("{0:20s}={1:8.5f}".format("ourad", self.ourad))
        print("{0:20s}={1:8.5f}".format("theta1", self.theta1))
        print("{0:20s}={1:8.5f}".format("theta2", self.theta2))
        print("{0:20s}={1:8.5f}".format("zeta", self.zeta))
        print()

    def plot(self, ax, screws=False):
        """
        Plot panel outline to ax
        Args:
            ax: matplotlib axes instance
            screws: Display screws in plot
        """
        x1 = self.inrad * np.sin(self.theta1)
        y1 = self.inrad * np.cos(self.theta1)
        x2 = self.ourad * np.sin(self.theta1)
        y2 = self.ourad * np.cos(self.theta1)
        ax.plot([x1, x2], [y1, y2], ls='-', color=self.linecolor, marker=None, lw=self.linewidth)
        if self.first:
            # Draw ring outline with first panel
            inrad = plt.Circle((0, 0), self.inrad, color=self.linecolor, fill=False, lw=self.linewidth)
            ourad = plt.Circle((0, 0), self.ourad, color=self.linecolor, fill=False, lw=self.linewidth)
            ax.add_patch(inrad)
            ax.add_patch(ourad)
        self.plot_label(ax)
        if screws:
            self.plot_screws(ax)
