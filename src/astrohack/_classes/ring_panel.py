import numpy as np
from matplotlib import pyplot as plt
from astrohack._classes.base_panel import BasePanel


class RingPanel(BasePanel):
    # This class describes and treats panels that are arranged in
    # rings on the Antenna surface

    def __init__(self, kind, angle, iring, ipanel, inrad, ourad, margin=0.05):
        """
        Initializes a panel that is a section of a ring in a circular antenna
        Args:
            kind: What kind of surface to be used in fitting ["rigid", "mean", "xyparaboloid",
            "rotatedparaboloid", "corotatedparaboloid"]
            angle: Azimuthal span of the panel
            iring: Which ring is the panel in
            ipanel: Panel number clockwise from top
            inrad: Radius at panel inner side
            ourad: Radius at panel outer side
            margin: Fraction from panel edge inwards that won't be used for fitting
        """
        self.iring = iring + 1
        self.inrad = inrad
        self.ourad = ourad
        self.theta1 = ipanel * angle
        self.theta2 = (ipanel + 1) * angle
        self.zeta = (ipanel + 0.5) * angle
        dradius = ourad-inrad
        self.margin_theta1 = self.theta1 + margin * angle
        self.margin_theta2 = self.theta2 - margin * angle
        self.margin_inrad = inrad + margin * dradius
        self.margin_ourad = ourad - margin * dradius
        self.solved = False

        screws = np.ndarray([4, 2])
        self.first = ipanel == 0
        rscale = 0.1 * dradius
        tscale = 0.1 * angle
        screws[0, :] = np.cos(self.theta1 + tscale), np.sin(self.theta1 + tscale)
        screws[1, :] = np.cos(self.theta2 - tscale), np.sin(self.theta2 - tscale)
        screws[2, :] = np.cos(self.theta1 + tscale), np.sin(self.theta1 + tscale)
        screws[3, :] = np.cos(self.theta2 - tscale), np.sin(self.theta2 - tscale)
        screws[0, :] *= (inrad + rscale)
        screws[1, :] *= (inrad + rscale)
        screws[2, :] *= (ourad - rscale)
        screws[3, :] *= (ourad - rscale)
        super().__init__(kind, screws, str(ipanel+1))

        rt = (self.inrad + self.ourad) / 2
        self.center = [rt * np.cos(self.zeta), rt * np.sin(self.zeta)]

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

    def export_adjustments(self, unit='mm'):
        """
        Exports panel screw adjustments to a string
        Args:
            unit: Unit for screw adjustments ['mm','miliinches']

        Returns:
        String with screw adjustments for this panel
        """
        string = '{0:8d} {1:8d}'.format(self.iring, self.label)
        return string+self.export_screw_adjustments(unit)

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
