import numpy as np
from shapely import Point, Polygon
from shapely.plotting import plot_polygon

from astrohack.antenna.base_panel import BasePanel
from astrohack.antenna.panel_fitting import PanelPoint
from astrohack.utils.constants import markersize
from astrohack.utils.text import spc


class PolygonPanel(BasePanel):
    def __init__(
        self,
        label,
        model,
        panel_info,
        panel_margin,
        plot_screw_size=0.1,
    ):
        """
        Initializes a polygon based panel based on a polygon shape and the screw positions
        Args:
            label: panel label
            model: What model of surface to be used in fitting ["rigid", "mean", "xyparaboloid", "rotatedparaboloid"]
            panel_info: Dictionary with panel information
        """
        if "corotated" in model:
            raise ValueError(
                f"corotated models such as {model} are not supported for Polygon based panels"
            )

        poly = Polygon(panel_info["polygon"])

        array_screws = panel_info["screws"]
        screw_list = []
        for screw in array_screws:
            screw_list.append(PanelPoint(screw[0], screw[1]))

        # Compute refence points for the flexible model, for Ring panels the flexible model uses 2 reference points
        # per ring. here we define reference points as the mean points of the lower and upper edges of the polygon
        # panels which are assumed to be approximately four sided.
        xcoords = []
        ycoords = []
        for corners in poly.exterior.coords:
            xcoords.append(corners[0])
            ycoords.append(corners[1])

        xcoords = np.array(xcoords)
        ycoords = np.array(ycoords)
        y_sort_idx = np.argsort(ycoords)
        y_ascending = ycoords[y_sort_idx]
        x_y_ascending = xcoords[y_sort_idx]
        y_jump = int(np.argmax(np.diff(y_ascending))) + 1
        x1 = np.mean(np.min(x_y_ascending[:y_jump]), np.max(x_y_ascending[:y_jump]))
        x2 = np.mean(np.min(x_y_ascending[y_jump:]), np.max(x_y_ascending[y_jump:]))
        y2 = np.mean(np.min(y_ascending[y_jump:]), np.max(y_ascending[y_jump:]))
        ref_points = [x1, x2, y2]

        screws = np.array(screw_list, dtype=np.object_)
        super().__init__(
            model,
            screws,
            screws,
            plot_screw_size,
            label,
            center=PanelPoint(poly.centroid.x, poly.centroid.y),
            zeta=None,
            ref_points=ref_points,
        )
        self.polygon = poly
        self.margin = panel_margin

        self.margin_polygon = self._init_margin_polygon()

        return

    def _init_margin_polygon(self):
        """
        This method assumes simple convex shapes for the polygons
        Returns:
            polygon separating margins from samples.
        """

        xc, yc = self.polygon.centroid.x, self.polygon.centroid.y
        center = np.array([xc, yc])
        corners = np.array(self.polygon.exterior.coords)

        diff = corners - center
        margin_corners = center + ((1 - self.margin) * diff[:])

        return Polygon(margin_corners)

    def is_inside(self, xc, yc):
        """
        Checks if a point is inside the panel by using shapely's point in polygon method
        Args:
            xc: point x coordinate
            yc: point y coordinate
        """
        inpanel = self.polygon.intersects(Point([xc, yc]))
        issample = self.margin_polygon.intersects(Point([xc, yc]))
        return issample, inpanel

    def print_misc(self, verbose=False):
        """
        Print miscelaneous information about the panel to the terminal
        Args:
            verbose: Include more information in print
        """
        print("########################################")
        print("{0:20s}={1:8d}".format("label", self.label))
        print("{0:20s}={1:8s}".format("model", spc + self.model_name))
        print("{0:20s}={1:8d}".format("nsamp", len(self.samples)))
        if verbose:
            for sample in self.samples:
                print(sample)
        print()

    def plot(self, ax, screws=False, label=False, margins=False, samples=False):
        """
        Plot panel outline to ax
        Args:
            ax: matplotlib axes instance
            screws: Display screws in plot
            label: add panel label to plot
            margins: plot margin polygon
            samples: plot samples
        """
        plot_polygon(
            self.polygon,
            ax=ax,
            add_points=False,
            color=self.linecolor,
            linewidth=self.linewidth,
            facecolor="none",
        )
        if margins:
            plot_polygon(
                self.margin_polygon,
                ax=ax,
                add_points=False,
                color=self.linecolor,
                linewidth=self.linewidth / 3,
                facecolor="none",
            )
        if samples and len(self.samples) > 0:
            for sample in self.samples:
                ax.scatter(sample.xc, sample.yc, color="black", s=markersize)
            for marg in self.margins:
                ax.scatter(marg.xc, marg.yc, color="red", s=markersize)
        if label:
            self.plot_label(ax, rotate=False)
        if screws:
            self.plot_screws(ax)
        return
