import numpy as np
from matplotlib import pyplot as plt
from astrohack._classes.base_panel import BasePanel


class RingPanel(BasePanel):
    # This class describes and treats panels that are arranged in
    # rings on the Antenna surface

    def __init__(self, kind, angle, ipanel, label, inrad, ourad, margin=0.05, screw_scheme=None):
        """
        Initializes a panel that is a section of a ring in a circular antenna
        Args:
            kind: What kind of surface to be used in fitting ["rigid", "mean", "xyparaboloid",
            "rotatedparaboloid", "corotatedparaboloid"]
            angle: Azimuthal span of the panel
            ipanel: Panel number clockwise from top
            label: Panel label
            inrad: Radius at panel inner side
            ourad: Radius at panel outer side
            margin: Fraction from panel edge inwards that won't be used for fitting
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
        screws = self._init_screws(screw_scheme)
        # Now we are ready to initialize the base object
        super().__init__(kind, screws, label, center=self.center, zeta=zeta)

    def _init_screws(self, scheme, offset=0.1):
        if scheme is None:
            scheme = ['il', 'ir', 'ol', 'or']
        nscrews = len(scheme)
        screws = np.ndarray([nscrews, 2])
        roffset = offset*(self.ourad-self.inrad)
        toffset = offset*(self.theta2-self.theta1)
        for iscrew in range(nscrews):
            if scheme[iscrew] == 'c':
                screws[iscrew, :] = self.center
                continue
            if scheme[iscrew][1] == 'l':
                screws[iscrew, :] = np.cos(self.theta1 + toffset), np.sin(self.theta1 + toffset)
            else:
                screws[iscrew, :] = np.cos(self.theta2 - toffset), np.sin(self.theta2 - toffset)
            if scheme[iscrew][0] == 'i':
                screws[iscrew, :] *= self.inrad + roffset
            else:
                screws[iscrew, :] *= self.ourad - roffset
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
