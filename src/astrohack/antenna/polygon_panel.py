from shapely import Polygon, Point
from shapely.plotting import plot_polygon
from astrohack.antenna.base_panel import BasePanel


class PolygonPanel(BasePanel):

    def __init__(self, label, model, panel_info, panel_margin, plot_screw_size=0.20,):
        """
        Initializes a polygon based panel based on a polygon shape and the screw positions
        Args:
            label: panel label
            model: What model of surface to be used in fitting ["rigid", "mean", "xyparaboloid", "rotatedparaboloid"]
            panel_info: Dictionary with panel information
        """
        if 'corotated' in model:
            raise Exception(
                f"corotated models such as {model} are not supported for Polygon based panels"
            )

        poly = Polygon(panel_info['polygon'])

        screws = panel_info['screws']
        super().__init__(model, screws, screws, plot_screw_size, label,
                         center=[poly.centroid.x, poly.centroid.y], zeta=None, ref_points=None,)
        self.polygon = poly
        self.margin = panel_margin

        if not self.polygon.is_simple:
            raise Exception("Polygon must not intersect itself")
        return

    def is_inside(self, point):
        """
        Checks if a point is inside the panel by using shapely's point in polygon method
        Args:
            point: point to be tested
        """
        return self.polygon.intersects(Point([point.xc, point.yc]))

    def print_misc(self, verbose=False):
        """
        Print miscelaneous information about the panel to the terminal
        Args:
            verbose: Include more information in print
        """
        print("########################################")
        print("{0:20s}={1:8d}".format("label", self.label))
        print("{0:20s}={1:8s}".format("model", " " + self.model_name))
        print("{0:20s}={1:8d}".format("nsamp", len(self.samples)))
        if verbose:
            for sample in self.samples:
                print(sample)
        print()

    def plot(self, ax, screws=False):
        """
        Plot panel outline to ax
        Args:
            ax: matplotlib axes instance
            screws: Display screws in plot
        """
        plot_polygon(
            self.polygon,
            ax=ax,
            add_points=False,
            color=self.linecolor,
            linewidth=self.linewidth,
        )
        self.plot_label(ax)
        if screws:
            self.plot_screws(ax)
        return
