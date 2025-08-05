import time

from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely.strtree import STRtree

import astrohack
import pathlib

import xarray as xr
import toolviper.utils.logger as logger

import astrohack.utils.tools
from astrohack.antenna.polygon_panel import PolygonPanel
from astrohack.utils.constants import *
from astrohack.antenna.ring_panel import RingPanel
from astrohack.utils.algorithms import create_coordinate_images, arm_shadow_masking
from astrohack.utils.ray_tracing_general import GlobalQPS


class Telescope:
    """
    Base telescope class containing IO methods and attributes that are common to all telescopes.
    """

    data_array_keys = ["point_cloud", "qps_coefficients"]
    data_array_dims = [["point_axis", "xyz"], ["qps_coefficients"]]
    excluded_keys = ["filepath", "filename", "z_cos_image"]

    def __init__(self):
        self.diameter = None
        self.antenna_list = None
        self.file_name = None
        self.file_path = None
        self.array_center = None
        self.comment = None
        self.inner_radial_limit = None
        self.outer_radial_limit = None
        self.name = None
        self.el_axis_offset = None
        self.station_distance_dict = None
        self.gain_wavelengths = None

    def read(self, filename):
        """
        Read the telescope object from an X array .zarr telescope configuration file
        Args:
            filename: name of the input file
        """

        try:
            logger.debug("Reading telescope data from: filename")
            xds = xr.open_zarr(filename)
            for key in xds.attrs:
                setattr(self, key, xds.attrs[key])

            for key in self.data_array_keys:
                try:
                    setattr(self, key, xds[key].values)
                except KeyError:
                    pass

        except FileNotFoundError:
            logger.error(f"Telescope file not found: {filename}")
            raise FileNotFoundError

        relative_path = pathlib.Path(filename)
        abs_path = relative_path.resolve()
        self.file_name = str(abs_path.name)
        self.file_path = str(abs_path.parent)

    def read_from_distro(self, name):
        """
        Read telescope info from files distributed with astrohack.
        Args:
            name: Name of the telescope to be read.

        Returns:
            None
        """
        dest_path = "/".join(
            [astrohack.__path__[0], f"data/telescopes/{name.lower()}.zarr"]
        )
        self.read(dest_path)

    def write(self, filename):
        """
        Write the telescope object to an X array .zarr telescope configuration file
        Args:
            filename: Name of the output file
        """
        obj_dict = vars(self)
        for key in self.excluded_keys:
            obj_dict.pop(key, None)

        xds = xr.Dataset()
        for ikey, key in enumerate(self.data_array_keys):
            try:
                data = obj_dict.pop(key)
                xds[key] = xr.DataArray(data, dims=self.data_array_dims[ikey])
            except KeyError:
                pass

        xds.attrs = obj_dict
        logger.debug("Writing telescope data to: filename")
        xds.to_zarr(filename, mode="w", compute=True, consolidated=True)
        return

    def write_to_distro(self):
        dest_path = "/".join(
            [
                astrohack.__path__[0],
                f'data/telescopes/{self.name.lower().replace(" ", "_")}.zarr',
            ]
        )
        self.write(dest_path)

    def __repr__(self):
        """
        Simple print function that prints all class attributes.
        Returns:
            String containing descriptions of all object attributes.
        """
        outstr = ""
        obj_dict = vars(self)
        for key, item in obj_dict.items():
            if isinstance(item, dict):
                key_list = list(item.keys())
                outstr += f"{key:20s} = dict({str(key_list)})\n"
            else:
                outstr += f"{key:20s} = {str(item)}\n"
        return outstr


class RingedCassegrain(Telescope):
    """
    Derived class containing description and methods pertaining to telescope whose panels are distributed in concentric
    rings from the dish center.
    """

    def __init__(self):
        super().__init__()

        self.panel_inner_radii = None
        self.panel_outer_radii = None
        self.focus = None
        self.arm_shadow_rotation = None
        self.arm_shadow_width = None
        self.magnification = None
        self.n_panel_per_ring = None
        self.panel_numbering = None
        self.screw_description = None
        self.screw_offset = None
        self.secondary_distance_to_focus = None
        self.secondary_support_shape = None
        self.surp_slope = None
        self.n_rings_of_panels = None

        self._panel_label = None

    @classmethod
    def from_name(cls, name):
        """
        Initialize and read from the distro a telescope object.
        Args:
            name: Name of the telescope to be read.

        Returns:
            RingedCassegrain object
        """
        obj = cls()
        obj.read_from_distro(name)
        return obj

    def consistency_check(self):
        """
        Make a simple check to test that some of its attributes are
        Returns:
            None
        """
        error = False

        if self.panel_outer_radii[-1] > self.diameter / 2.0:
            logger.error("Panel description goes beyond dish outermost radius")
            error = True

        if (
            not self.n_rings_of_panels
            == len(self.panel_inner_radii)
            == len(self.panel_outer_radii)
        ):
            logger.error(
                "Number of panels don't match radii or number of panels list sizes"
            )
            error = True

        if error:
            raise Exception("Failed Consistency check")
        else:
            print("Consistency passed")

        return

    @staticmethod
    def _vla_panel_labeling(iring, ipanel):
        """
        Provide the correct panel label for VLA style panels
        Args:
            iring: Number of the ring the panel is in
            ipanel: Number of the panel in that ring clockwise from the top
        Returns:
            The proper label for the panel at iring, ipanel
        """
        return "{0:d}-{1:d}".format(iring + 1, ipanel + 1)

    def _alma_panel_labeling(self, iring, ipanel):
        """
        Provide the correct panel label for ALMA style panels, which is more complicated than VLA panels due to the
        implementation of panel sectors
        Args:
            iring: Number of the ring the panel is in
            ipanel: Number of the panel in that ring clockwise from the top

        Returns:
            The proper label for the panel at iring, ipanel
        """
        angle = twopi / self.n_panel_per_ring[iring]
        sector_angle = twopi / self.n_panel_per_ring[0]
        theta = twopi - (ipanel + 0.5) * angle
        sector = int(
            ((theta / sector_angle) + 1 + self.n_panel_per_ring[0] / 4)
            % self.n_panel_per_ring[0]
        )
        if sector == 0:
            sector = self.n_panel_per_ring[0]
        nppersec = self.n_panel_per_ring[iring] / self.n_panel_per_ring[0]
        jpanel = int(nppersec - (ipanel % nppersec))
        return "{0:1d}-{1:1d}{2:1d}".format(sector, iring + 1, jpanel)

    def build_panel_list(self, panel_model, panel_margins):
        """
        Construct a list of panel objects according to the telescope description
        Args:
            panel_model: Type of panel model to be fitted.
            panel_margins: how much of the panel

        Returns:
            List containing RingPanel objects
        """

        if self.name in ["VLA", "VLBA"]:
            self._panel_label = self._vla_panel_labeling
        elif "ALMA" in self.name or self.name == "ACA 7m":
            self._panel_label = self._alma_panel_labeling
        else:
            raise Exception(f"Don't know how to build panel list for {self.name}")

        panel_list = []
        for iring in range(self.n_rings_of_panels):
            angle = twopi / self.n_panel_per_ring[iring]
            for ipanel in range(self.n_panel_per_ring[iring]):
                panel = RingPanel(
                    panel_model,
                    angle,
                    ipanel,
                    self._panel_label(iring, ipanel),
                    self.panel_inner_radii[iring],
                    self.panel_outer_radii[iring],
                    margin=panel_margins,
                    screw_scheme=self.screw_description,
                    screw_offset=self.screw_offset,
                    plot_screw_size=0.006 * self.diameter,
                )
                panel_list.append(panel)

        return panel_list

    def attribute_pixels_to_panels(
        self, panel_list, u_axis, v_axis, radius, phi, deviation, mask
    ):
        """
        Attribute pixels in deviation image to the panels in the panel_list
        Args:
            panel_list: The panel list must have been created by build_panel_list for the same instrument
            u_axis: Aperture U axis
            v_axis: Aperture V axis
            radius: Aperture radius map
            phi: Aperture phi angle map
            deviation: Aperture deviation
            mask: Aperture mask

        Returns:
            map of panel attributions
        """
        panel_map = np.full_like(radius, -1)
        panelsum = 0
        for iring in range(self.n_rings_of_panels):
            angle = twopi / self.n_panel_per_ring[iring]
            panel_map = np.where(
                radius >= self.panel_inner_radii[iring],
                np.floor(phi / angle) + panelsum,
                panel_map,
            )
            panelsum += self.n_panel_per_ring[iring]
        panel_map = np.where(radius >= self.panel_inner_radii[0], panel_map, np.nan)
        panel_map = np.where(radius > self.panel_outer_radii[-1], np.nan, panel_map)
        u_mesh, v_mesh = create_coordinate_images(u_axis, v_axis)

        for ix, xc in enumerate(u_axis):
            for iy, yc in enumerate(v_axis):
                ipanel = panel_map[ix, iy]
                if ipanel >= 0:
                    xc = u_mesh[ix, iy]
                    yc = v_mesh[ix, iy]
                    panel = panel_list[int(ipanel)]
                    issample, inpanel = panel.is_inside(radius[ix, iy], phi[ix, iy])
                    if inpanel:
                        if issample and mask[ix, iy]:
                            panel.add_sample([xc, yc, ix, iy, deviation[ix, iy]])
                        else:
                            panel.add_margin([xc, yc, ix, iy, deviation[ix, iy]])

        return panel_map

    def create_aperture_mask(
        self,
        u_axis,
        v_axis,
        use_detailed_mask=True,
        return_polar_meshes=False,
        use_outer_limit=False,
    ):
        u_mesh, v_mesh, radius_mesh, polar_angle_mesh = create_coordinate_images(
            u_axis, v_axis, create_polar_coordinates=True
        )

        if use_outer_limit:
            outer_radius = self.outer_radial_limit
        else:
            outer_radius = self.diameter / 2.0

        mask = np.full_like(radius_mesh, True, dtype=bool)
        mask = np.where(radius_mesh > outer_radius, False, mask)
        mask = np.where(radius_mesh < self.inner_radial_limit, False, mask)

        if self.arm_shadow_width is None or not use_detailed_mask:
            pass
        elif isinstance(self.arm_shadow_width, (float, int)):
            mask = arm_shadow_masking(
                mask,
                u_mesh,
                v_mesh,
                radius_mesh,
                self.inner_radial_limit,
                outer_radius,
                self.arm_shadow_width,
                self.arm_shadow_rotation,
            )
        elif isinstance(self.arm_shadow_width, list):
            for section in self.arm_shadow_width:
                minradius, maxradius, width = section
                mask = arm_shadow_masking(
                    mask,
                    u_mesh,
                    v_mesh,
                    radius_mesh,
                    minradius,
                    maxradius,
                    width,
                    self.arm_shadow_rotation,
                )

        else:
            raise Exception(
                f"Don't know how to handle an arm width of class {type(self.arm_shadow_width)}"
            )

        if return_polar_meshes:
            return mask, radius_mesh, polar_angle_mesh
        else:
            return mask

    def phase_to_deviation(self, u_axis, v_axis, mask, phase, wavelength):
        """
        Transform phase image to physical deviation image based on wavelength.
        Args:
            u_axis: Aperture U axis
            v_axis: Aperture V axis
            mask: dummy argument for interface compatibility
            phase: Phase image in Radians
            wavelength: Observation wavelength in meters

        Returns:
            Deviation image.
        """
        _, _, radius, _ = create_coordinate_images(
            u_axis, v_axis, create_polar_coordinates=True
        )
        acoeff = (wavelength / twopi) / (4.0 * self.focus)
        bcoeff = 4 * self.focus**2
        return acoeff * phase * np.sqrt(radius**2 + bcoeff)

    def deviation_to_phase(self, u_axis, v_axis, mask, deviation, wavelength):
        """ "
        Transform deviation image to physical phase image based on wavelength.
        Args:
            u_axis: Aperture U axis
            v_axis: Aperture V axis
            mask: dummy argument for interface compatibility
            deviation: Deviation image in meters
            wavelength: Observation wavelength in meters

        Returns:
            Phase image.
        """
        _, _, radius, _ = create_coordinate_images(
            u_axis, v_axis, create_polar_coordinates=True
        )
        acoeff = (wavelength / twopi) / (4.0 * self.focus)
        bcoeff = 4 * self.focus**2
        return deviation / (acoeff * np.sqrt(radius**2 + bcoeff))


class NgvlaPrototype(Telescope):
    """
    Derived class to contain ngVLA prototype specific methods and attributes.
    """

    def __init__(self):
        super().__init__()
        self.panel_dict = None
        self.screw_description = None
        self.point_cloud = None
        self.qps_coefficients = None
        self.aperture_polygon = None

        # This is not to be written to disk
        self.z_cos_image = None

    @classmethod
    def from_name(cls, name):
        obj = cls()
        obj.read_from_distro(name)
        return obj

    def build_panel_list(self, panel_model, panel_margins):
        panel_list = []
        for panel_label, panel_info in self.panel_dict.items():
            panel = PolygonPanel(panel_label, panel_model, panel_info, panel_margins)
            panel_list.append(panel)
        return panel_list

    @staticmethod
    def attribute_pixels_to_panels(
        panel_list, u_axis, v_axis, radius, _, deviation, mask
    ):
        u_mesh, v_mesh = create_coordinate_images(u_axis, v_axis)
        panel_map = np.full_like(radius, np.nan)
        for ix, xc in enumerate(u_axis):
            for iy, yc in enumerate(v_axis):
                if mask[ix, iy]:
                    xc = u_mesh[ix, iy]
                    yc = v_mesh[ix, iy]
                    for ipanel, panel in enumerate(panel_list):
                        issample, inpanel = panel.is_inside(xc, yc)
                        if inpanel:
                            if issample:
                                panel.add_sample([xc, yc, ix, iy, deviation[ix, iy]])
                            else:
                                panel.add_margin([xc, yc, ix, iy, deviation[ix, iy]])
                            panel_map[ix, iy] = ipanel
                            break

        return panel_map

    def create_aperture_mask(
        self,
        u_axis,
        v_axis,
        use_detailed_mask=True,
        return_polar_meshes=False,
        use_outer_limit=False,
    ):
        u_mesh, v_mesh, radius_mesh, polar_angle_mesh = create_coordinate_images(
            u_axis, v_axis, create_polar_coordinates=True
        )
        if use_detailed_mask:
            # This is the slowest line, means of optimizing this would be great
            point_list = [Point(u_val, -v_val) for v_val in v_axis for u_val in u_axis]
            pnt_str_tree = STRtree(point_list)
            ap_polygon = Polygon(self.aperture_polygon)
            intersection = pnt_str_tree.query(ap_polygon, predicate="intersects")
            mask_1d = np.full((v_axis.shape[0] * u_axis.shape[0]), False)
            mask_1d[intersection] = True
            mask = mask_1d.reshape((u_axis.shape[0], v_axis.shape[0]))
        else:
            if use_outer_limit:
                outer_radius = self.outer_radial_limit
            else:
                outer_radius = self.diameter / 2.0

            mask = np.full_like(radius_mesh, True, dtype=bool)
            mask = np.where(radius_mesh > outer_radius, False, mask)
            # This line does not need to be included as there is no blockage in the ngvla!
            # mask = np.where(radius_mesh < self.inner_radial_limit, False, mask)

        if return_polar_meshes:
            return mask, radius_mesh, polar_angle_mesh
        else:
            return mask

    def phase_to_deviation(self, u_axis, v_axis, mask, phase, wavelength):
        if self.z_cos_image is None:
            global_qps = GlobalQPS.from_point_cloud_and_coefficients(
                self.point_cloud, self.qps_coefficients
            )
            self.z_cos_image = global_qps.compute_gridded_z_cos(u_axis, v_axis, mask)

        deviation = phase * wavelength / fourpi / self.z_cos_image
        return deviation

    def deviation_to_phase(self, u_axis, v_axis, mask, deviation, wavelength):
        if self.z_cos_image is None:
            global_qps = GlobalQPS.from_point_cloud_and_coefficients(
                self.point_cloud, self.qps_coefficients
            )
            self.z_cos_image = global_qps.compute_gridded_z_cos(u_axis, v_axis, mask)

        phase = fourpi * self.z_cos_image * deviation / wavelength
        return phase


def get_proper_telescope(name: str, antenna_name: str = None):
    """
    Retrieve the proper telescope object based on the name
    Args:
        name: Name of the telescope
        antenna_name: Name of the antenna, significant for heterogenius arrays.
    Returns:
        A telescope object of one of the proper subclasses
    """

    name = name.lower()
    if isinstance(antenna_name, str):
        antenna_name = antenna_name.lower()
    if "ngvla" in name:
        return NgvlaPrototype.from_name("ngvla_proto_2025")
    elif "vla" in name:
        if antenna_name is None or "ea" in antenna_name or antenna_name == "all":
            return RingedCassegrain.from_name("vla")
        elif "na" in antenna_name:
            return NgvlaPrototype.from_name("ngvla_proto_2025")
        else:
            raise Exception(f"Unsupported antenna type for the VLA: {antenna_name}")

    elif "vlba" in name:
        return RingedCassegrain.from_name("vlba")

    elif "alma" in name:
        if antenna_name is None:
            raise Exception(
                "ALMA is an heterogenious array and hence an antenna name is needed"
            )
        elif "dv" in antenna_name:
            return RingedCassegrain.from_name("alma_dv")
        elif "da" in antenna_name:
            return RingedCassegrain.from_name("alma_da")
        elif "tp" in antenna_name:
            return RingedCassegrain.from_name("alma_tp")
        else:
            raise Exception(f"Unsupported antenna type for ALMA: {antenna_name}")

    elif "aca" in name:
        return RingedCassegrain.from_name("aca_7m")

    else:
        return None
