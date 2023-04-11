import pytest
import shapely
from astrohack._classes.polygon_panel import PolygonPanel
import numpy as np


class TestPolygonPanel:
    # A simple square
    polygon = np.array([(0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)])
    position = 0
    screws = np.zeros([4, 2])
    xyparapan = PolygonPanel('xyparaboloid', position, polygon, screws)

    def test_init(self):
        """
        Tests the initialization of a PolygonPanel object
        """
        assumedcenter = [0.5, 0.5]
        assert isinstance(self.xyparapan.polygon, shapely.Polygon), 'Polygon is not correctly initialized'
        assert self.xyparapan.center[0] == assumedcenter[0], 'Polygon x center is incorrect'
        assert self.xyparapan.center[1] == assumedcenter[1], 'Polygon y center is incorrect'
        with pytest.raises(Exception):
            lepanel = PolygonPanel('corotatedparaboloid', self.position, self.polygon, self.screws)
        return

    def test_is_inside(self):
        """
        Tests the is_inside method by testing points that have known positions relative to the panel
        """
        apointinside = [0.25, 0.75]
        apointoutside = [-0.25, 1.0]
        apointontheborder = [0.5, 1.0]

        assert self.xyparapan.is_inside(apointinside)
        assert self.xyparapan.is_inside(apointontheborder)
        assert not self.xyparapan.is_inside(apointoutside)
        return

    def test_export_adjustments(self):
        """
        Tests that Panel numbering is correctly added to the exported string
        """
        point = [0.5, 0.5, 1, 1, 2.5]
        npoints = 200
        for i in range(npoints):
            self.xyparapan.add_point(point)
        self.xyparapan.solve()
        self.xyparapan.get_corrections()
        exportedstr = self.xyparapan.export_adjustments().split()

        assert float(exportedstr[0]) == self.position + 1, 'Panel numbering is wrong when exported'
        return
