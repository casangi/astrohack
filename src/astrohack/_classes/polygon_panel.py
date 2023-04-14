from shapely import Polygon, Point
from shapely.plotting import plot_polygon
from astrohack._classes.base_panel import BasePanel, panel_models, icorpara


class PolygonPanel(BasePanel):

    def __init__(self, model, ipanel, polygon, screws):
        """
        Initializes a polygon based panel based on a polygon shape and the screw positions
        Args:
            model: What model of surface to be used in fitting ["rigid", "mean", "xyparaboloid", "rotatedparaboloid"]
            ipanel: Panel number
            polygon: Polygon describing the panel shape
            screws: Positions of the screw over the panel
        """
        if model == panel_models[icorpara]:
            raise Exception('corotatedparaboloid not supported for Polygon based panels')
        super().__init__(model, ipanel, screws)
        self.polygon = Polygon(polygon)
        if not self.polygon.is_simple:
            raise Exception('Polygon must not intersect itself')
        self.center = self.polygon.centroid.x, self.polygon.centroid.y
        return

    def is_inside(self, point):
        """
        Checks if a point is inside the panel by using shapely's point in polygon method
        Args:
            point: point to be tested
        """
        return self.polygon.intersects(Point(point))

    def export_adjustments(self, unit='mm'):
        """
        Exports panel screw adjustments to a string
        Args:
            unit: Unit for screw adjustments ['mm','miliinches']

        Returns:
        String with screw adjustments for this panel
        """
        string = '{0:8d}'.format(self.ipanel)
        return string+self.export_screw_adjustments(unit)

    def print_misc(self, verbose=False):
        """
        Print miscelaneous information about the panel to the terminal
        Args:
            verbose: Include more information in print
        """
        print("########################################")
        print("{0:20s}={1:8d}".format("ipanel", self.ipanel))
        print("{0:20s}={1:8s}".format("model", " " + self.model))
        print("{0:20s}={1:8d}".format("nsamp", self.nsamp))
        if verbose:
            for isamp in range(self.nsamp):
                strg = "{0:20s}=".format("samp{0:d}".format(isamp))
                for val in self.values[isamp]:
                    strg += str(val) + ", "
                print(strg)
        print()

    def plot(self, ax, screws=False):
        """
        Plot panel outline to ax
        Args:
            ax: matplotlib axes instance
            screws: Display screws in plot
        """
        plot_polygon(self.polygon, ax=ax, add_points=False, color=self.linecolor, linewidth=self.linewidth)
        self.plot_label(ax)
        if screws:
            self.plot_screws(ax)
        return
