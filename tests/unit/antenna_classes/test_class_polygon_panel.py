import numpy as np
import pytest
from shapely import Polygon

from astrohack.antenna.polygon_panel import PolygonPanel


class TestPolygonPanel:
    polygon = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    screws = [[0.05, 0.05], [0.05, 0.95], [0.95, 0.95], [0.95, 0.05]]
    label = "test"
    model = "rigid"
    panel_info = {"polygon": polygon, "screws": screws}
    panel_margin = 0.2
    deviation = 2.0
    point = [2.5, -2.5, 1, 1, deviation]
    panel = PolygonPanel(label, model, panel_info, panel_margin)

    def test_init(self):
        """
        Tests the correct initialization of a PolygonPanel object, not all parameters tested
        """

        assert isinstance(self.panel.polygon, Polygon), (
            "Polygon is not an instance of a shapely polygon"
        )
        assert isinstance(self.panel.margin_polygon, Polygon), (
            "Margin Polygon is not an instance of a shapely polygon"
        )

        margin_poly = self.panel.margin_polygon.exterior.coords

        margin_reference = [
            [
                0.1,
                0.1,
            ],
            [0.1, 0.9],
            [0.9, 0.9],
            [0.9, 0.1],
            [
                0.1,
                0.1,
            ],
        ]

        assert np.all(np.isclose(margin_poly, margin_reference))

        with pytest.raises(ValueError):
            PolygonPanel(
                self.label, "corotated_paraboloid", self.panel_info, self.panel_margin
            )

    def test_is_inside(self):
        """
        Test over the is_inside test for a point
        """
        is_sample, is_in_panel = self.panel.is_inside(0.5, 0.5)
        assert is_sample and is_in_panel, (
            "center of the panel must be a sample and inside panel"
        )

        is_sample, is_in_panel = self.panel.is_inside(2.0, 2.0)
        assert not is_sample and not is_in_panel, (
            "Point outside panel must not be inside panel"
        )

        is_sample, is_in_panel = self.panel.is_inside(0.95, 0.95)
        assert not is_sample and is_in_panel, (
            "Point at margin must be inside but not a sample"
        )
