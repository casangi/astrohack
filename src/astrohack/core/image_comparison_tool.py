import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import xarray as xr
import pathlib

from astrohack.antenna.telescope import Telescope
from astrohack.utils.text import statistics_to_text
from astrohack.utils.algorithms import (
    create_aperture_mask,
    data_statistics,
    are_axes_equal,
)
from astrohack.visualization.plot_tools import well_positioned_colorbar, compute_extent
from astrohack.visualization.plot_tools import (
    close_figure,
    get_proper_color_map,
    scatter_plot,
)
from astrohack.utils.fits import (
    read_fits,
    get_axis_from_fits_header,
    get_stokes_axis_iaxis,
    put_axis_in_fits_header,
    write_fits,
)


def test_image(fits_image):
    if isinstance(fits_image, FITSImage):
        pass
    else:
        raise TypeError("Reference image is not a FITSImage object")


class FITSImage:

    def __init__(self):
        """
        Blank slate initialization of the FITSImage object
        """
        # Attributes:
        self.filename = None
        self.telescope_name = None
        self.rootname = None
        self.factor = 1.0
        self.reference_name = None
        self.resampled = False

        # Metadata
        self.header = None
        self.unit = None
        self.x_axis = None
        self.y_axis = None
        self.original_x_axis = None
        self.original_y_axis = None
        self.x_unit = None
        self.y_unit = None

        # Data variables
        self.original_data = None
        self.data = None
        self.residuals = None
        self.residuals_percent = None
        self.divided_image = None

    @classmethod
    def from_xds(cls, xds):
        """
        Initialize a FITSImage object using as a base a Xarray dataset
        Args:
            xds: Xarray dataset

        Returns:
            FITSImage object initialized from a xds
        """
        return_obj = cls()
        return_obj._init_as_xds(xds)
        return return_obj

    @classmethod
    def from_fits_file(cls, fits_filename, telescope_name):
        """
        Initialize a FITSImage object using as a base a FITS file.
        Args:
            fits_filename: FITS file on disk
            telescope_name: Name of the telescope used

        Returns:
            FITSImage object initialized from a FITS file
        """
        return_obj = cls()
        return_obj._init_as_fits(fits_filename, telescope_name)
        return return_obj

    @classmethod
    def from_zarr(cls, zarr_filename):
        """
        Initialize a FITSImage object using as a base a Xarray dataset store on disk in a zarr container
        Args:
            zarr_filename: Xarray dataset on disk as a zarr container

        Returns:
            FITSImage object initialized from a xds
        """
        return_obj = cls()
        xds = xr.open_zarr(zarr_filename)
        return_obj._init_as_xds(xds)
        return return_obj

    def _init_as_fits(self, fits_filename, telescope_name, istokes=0, ichan=0):
        """
        Backend for FITSImage.from_fits_file
        Args:
            fits_filename: FITS file on disk
            telescope_name: Name of the telescope used
            istokes: Stokes axis element to be fetched, should always be zero (singleton stokes axis or fetching I)
            ichan: Channel axis element to be fetched, should be zero for most cases, unless image has multiple channels

        Returns:
            None
        """
        self.filename = fits_filename
        self.telescope_name = telescope_name
        self.rootname = ".".join(fits_filename.split(".")[:-1]) + "."
        self.header, self.data = read_fits(self.filename, header_as_dict=True)
        stokes_iaxis = get_stokes_axis_iaxis(self.header)

        self.unit = self.header["BUNIT"]

        if len(self.data.shape) == 4:
            if stokes_iaxis == 4:
                self.data = self.data[istokes, ichan, ...]
            else:
                self.data = self.data[ichan, istokes, ...]
        elif len(self.data.shape) == 2:
            pass  # image is already as expected
        else:
            raise Exception(f"FITS image has an unsupported shape: {self.data.shape}")

        self.original_data = np.copy(self.data)

        if "AIPS" in self.header["ORIGIN"]:
            self.x_axis, _, self.x_unit = get_axis_from_fits_header(
                self.header, 1, pixel_offset=False
            )
            self.y_axis, _, self.y_unit = get_axis_from_fits_header(
                self.header, 2, pixel_offset=False
            )
            self.x_unit = "m"
            self.y_unit = "m"
        elif "Astrohack" in self.header["ORIGIN"]:
            self.x_axis, _, self.x_unit = get_axis_from_fits_header(self.header, 1)
            self.y_axis, _, self.y_unit = get_axis_from_fits_header(self.header, 2)
            self.data = np.fliplr(self.data)
        else:
            raise Exception(f'Unrecognized origin:\n{self.header["origin"]}')
        self._create_base_mask()
        self.original_x_axis = np.copy(self.x_axis)
        self.original_y_axis = np.copy(self.y_axis)

    def _init_as_xds(self, xds):
        """
        Backend for FITSImage.from_xds
        Args:
            xds: Xarray DataSet
        Returns:
            None
        """
        for key in xds.attrs:
            setattr(self, key, xds.attrs[key])

        self.x_axis = xds.x.values
        self.y_axis = xds.y.values
        self.original_x_axis = xds.original_x.values
        self.original_y_axis = xds.original_y.values

        for key, value in xds.items():
            setattr(self, str(key), xds[key].values)

    def _create_base_mask(self):
        """
        Create a base mask based on telescope parameters such as arm shadows.
        Returns:
            None
        """
        telescope_obj = Telescope(self.telescope_name)
        self.base_mask = create_aperture_mask(
            self.x_axis,
            self.y_axis,
            telescope_obj.inlim,
            telescope_obj.oulim,
            arm_width=telescope_obj.arm_shadow_width,
            arm_angle=telescope_obj.arm_shadow_rotation,
        )

    def resample(self, ref_image):
        """
        Resamples the data on this object onto the grid in ref_image
        Args:
            ref_image: Reference FITSImage object

        Returns:
            None
        """
        test_image(ref_image)
        x_mesh_orig, y_mesh_orig = np.meshgrid(self.x_axis, self.y_axis, indexing="ij")
        x_mesh_dest, y_mesh_dest = np.meshgrid(
            ref_image.x_axis, ref_image.y_axis, indexing="ij"
        )
        resamp = griddata(
            (x_mesh_orig.ravel(), y_mesh_orig.ravel()),
            self.data.ravel(),
            (x_mesh_dest.ravel(), y_mesh_dest.ravel()),
            method="linear",
        )
        size = ref_image.x_axis.shape[0], ref_image.y_axis.shape[0]
        self.x_axis = ref_image.x_axis
        self.y_axis = ref_image.y_axis
        self.data = resamp.reshape(size)
        self._create_base_mask()
        self.resampled = True

    def compare_difference(self, ref_image):
        """
        Does the difference comparison between self and ref_image.
        Args:
            ref_image: Reference FITSImage object

        Returns:
            None
        """
        test_image(ref_image)
        if not self.image_has_same_sampling(ref_image):
            self.resample(ref_image)

        self.residuals = ref_image.data - (self.data * self.factor)
        self.residuals_percent = 100 * self.residuals / ref_image.data
        self.reference_name = ref_image.filename

    def compare_scaled_difference(self, ref_image, rejection=10):
        """
        Does the scaled difference comparison between self and ref_image.
        Args:
            ref_image: Reference FITSImage object
            rejection: rejection level for scaling factor

        Returns:
            None
        """
        test_image(ref_image)
        if not self.image_has_same_sampling(ref_image):
            self.resample(ref_image)
        simple_division = ref_image.data / self.data
        rough_factor = np.nanmean(simple_division[self.base_mask])
        self.divided_image = np.where(
            np.abs(simple_division) > rejection * rough_factor, np.nan, simple_division
        )
        self.factor = np.nanmedian(self.divided_image)
        self.compare_difference(ref_image)

    def image_has_same_sampling(self, ref_image):
        """
        Tests if self has the same X and Y sampling as ref_image
        Args:
            ref_image: Reference FITSImage object

        Returns:
            True or False
        """
        test_image(ref_image)
        return are_axes_equal(self.x_axis, ref_image.x_axis) and are_axes_equal(
            self.y_axis, ref_image.y_axis
        )

    def mask_array(self, image_array):
        """
        Applies base mask to image_array
        Args:
            image_array: Data array to be masked

        Returns:
            Masked array
        """
        return np.where(self.base_mask, image_array, np.nan)

    def mask_original(self):
        """
        Applies base mask equivalent to original data
        Returns:
            Masked original data
        """
        telescope_obj = Telescope(self.telescope_name)
        orig_mask = create_aperture_mask(
            self.original_x_axis,
            self.original_y_axis,
            telescope_obj.inlim,
            telescope_obj.oulim,
            arm_width=telescope_obj.arm_shadow_width,
            arm_angle=telescope_obj.arm_shadow_rotation,
        )
        return np.where(orig_mask, self.original_data, np.nan)

    def plot_images(
        self,
        destination,
        ref_image,
        plot_resampled=False,
        plot_percentuals=False,
        plot_reference=False,
        plot_original=False,
        plot_divided_image=False,
        colormap="viridis",
        dpi=300,
        display=False,
    ):
        """
        Plot image contents of the FITSImage object, always plots the residuals when called
        Args:
            destination: Location onto which save plot files
            ref_image: reference image
            plot_resampled: Also plot data array?
            plot_percentuals: Also plot percentual residuals array?
            plot_reference: Also plot reference image?
            plot_original: Also plot original unresampled image?
            plot_divided_image: Also plot divided image?
            colormap: Colormap name for image plots
            dpi: png resolution on disk
            display: Show interactive view of plots

        Returns:
            None
        """

        extent = compute_extent(self.x_axis, self.y_axis, 0.0)
        cmap = get_proper_color_map(colormap)
        base_name = f"{destination}/{self.rootname}"

        if self.residuals is None:
            raise Exception("Cannot plot results as they don't exist yet.")
        self._plot_map(
            self.mask_array(self.residuals),
            f"Residuals, {self.reference_name} - {self.filename}",
            f"Residuals [{self.unit}]",
            f"{base_name}residuals.png",
            cmap,
            extent,
            "symmetrical",
            dpi,
            display,
            add_statistics=True,
        )

        if plot_resampled:
            self._plot_map(
                self.mask_array(self.data),
                "Resampled Data",
                f"Data [{self.unit}]",
                f"{base_name}resampled.png",
                cmap,
                extent,
                [None, None],
                dpi,
                display,
                add_statistics=True,
            )

        if plot_reference:
            self._plot_map(
                self.mask_array(ref_image.data),
                f"Reference: {self.reference_name}",
                f"Data [{self.unit}]",
                f"{base_name}reference.png",
                cmap,
                extent,
                [None, None],
                dpi,
                display,
                add_statistics=True,
            )
        if plot_original:
            self._plot_map(
                self.mask_original(),
                f"Unresampled data",
                f"Data [{self.unit}]",
                f"{base_name}original.png",
                cmap,
                extent,
                [None, None],
                dpi,
                display,
                add_statistics=True,
            )

        if plot_percentuals:
            if self.residuals is None:
                raise Exception("Cannot plot results as they don't exist yet.")
            self._plot_map(
                self.mask_array(self.residuals_percent),
                f"Residuals in %, {self.reference_name} - {self.filename}",
                f"Residuals [%]",
                f"{base_name}residuals_percent.png",
                cmap,
                extent,
                "symmetrical",
                dpi,
                display,
                add_statistics=True,
            )

        if plot_divided_image:
            if self.divided_image is None:
                pass
            else:
                self._plot_map(
                    self.mask_array(self.divided_image),
                    f"Divided image, {self.reference_name} / {self.filename}, scaling={self.factor:.4f}",
                    f"Division [ ]",
                    f"{base_name}divided.png",
                    cmap,
                    extent,
                    [None, None],
                    dpi,
                    display,
                    add_statistics=True,
                )

    def _plot_map(
        self,
        data,
        title,
        zlabel,
        filename,
        cmap,
        extent,
        zscale,
        dpi,
        display,
        add_statistics=False,
    ):
        """
        Backend for plot_images
        Args:
            data: Data array to be plotted
            title: Title to appear on plot
            zlabel: Label for the colorbar
            filename: name for the png file on disk
            cmap: Colormap object for plots
            extent: extents of the X and Y axes
            zscale: Constraints on the Z axes.
            dpi: png resolution on disk
            display: Show interactive view of plots
            add_statistics: Add simple statistics to plot's subtitle

        Returns:
            None
        """
        fig, ax = plt.subplots(1, 1, figsize=[10, 8])
        if zscale == "symmetrical":
            scale = max(np.abs(np.nanmin(data)), np.abs(np.nanmax(data)))
            vmin, vmax = -scale, scale
        else:
            vmin, vmax = zscale
            if vmin == "None" or vmin is None:
                vmin = np.nanmin(data)
            if vmax == "None" or vmax is None:
                vmax = np.nanmax(data)

        im = ax.imshow(
            data,
            cmap=cmap,
            interpolation="nearest",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        well_positioned_colorbar(
            ax, fig, im, zlabel, location="right", size="5%", pad=0.05
        )
        ax.set_xlabel(f"X axis [{self.x_unit}]")
        ax.set_ylabel(f"Y axis [{self.y_unit}]")
        if add_statistics:
            data_stats = data_statistics(data)
            ax.set_title(statistics_to_text(data_stats, num_format="dynamic"))
        close_figure(fig, title, filename, dpi, display)

    def export_as_xds(self):
        """
        Create a Xarray DataSet from the FITSImage object
        Returns:
            Xarray DataSet
        """
        xds = xr.Dataset()
        obj_dict = vars(self)

        coords = {
            "x": self.x_axis,
            "y": self.y_axis,
            "original_x": self.original_x_axis,
            "original_y": self.original_y_axis,
        }
        for key, value in obj_dict.items():
            failed = False
            if isinstance(value, np.ndarray):
                if len(value.shape) == 2:
                    if "original" in key:
                        xds[key] = xr.DataArray(
                            value, dims=["original_x", "original_y"]
                        )
                    else:
                        xds[key] = xr.DataArray(value, dims=["x", "y"])
                elif len(value.shape) == 1:
                    pass  # Axes
                else:
                    failed = True
            else:
                xds.attrs[key] = value

            if failed:
                raise Exception(f"Don't know what to do with: {key}")

        xds = xds.assign_coords(coords)
        return xds

    def to_zarr(self, zarr_filename):
        """
        Saves a xds representation of self on disk using the zarr format.
        Args:
            zarr_filename: Name for the zarr container on disk

        Returns:
            None
        """
        xds = self.export_as_xds()
        xds.to_zarr(zarr_filename, mode="w", compute=True, consolidated=True)

    def __repr__(self):
        """
        Print method
        Returns:
            A String summary of the current status of self.
        """
        obj_dict = vars(self)
        outstr = ""
        for key, value in obj_dict.items():
            if isinstance(value, np.ndarray):
                outstr += f"{key:17s} -> {value.shape}"
            elif isinstance(value, dict):
                outstr += f"{key:17s} -> dict()"
            else:
                outstr += f"{key:17s} =  {value}"
            outstr += "\n"
        return outstr

    def export_to_fits(self, destination):
        """
        Export internal images to FITS files.
        Args:
            destination: location to store FITS files

        Returns:
            None
        """
        pathlib.Path(destination).mkdir(exist_ok=True)
        ext_fits = ".fits"
        out_header = self.header.copy()

        put_axis_in_fits_header(out_header, self.x_axis, 1, "", self.x_unit)
        put_axis_in_fits_header(out_header, self.y_axis, 2, "", self.y_unit)

        obj_dict = vars(self)
        for key, value in obj_dict.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) == 2:
                    if "original" in key:
                        pass
                    else:
                        if key == "base_mask" or key == "divided_image":
                            unit = ""

                        elif key == "residuals_percent":
                            unit = "%"
                        else:
                            unit = self.unit
                        filename = f"{destination}/{self.rootname}{key}{ext_fits}"
                        write_fits(
                            out_header,
                            key,
                            np.fliplr(value.astype(float)),
                            filename,
                            unit,
                            reorder_axis=False,
                        )

    def scatter_plot(self, destination, ref_image, dpi=300, display=False):
        """
        Produce a scatter plot of self.data agains ref_image.data
        Args:
            destination: Location to store scatter plot
            ref_image: Reference FITSImage object
            dpi: png resolution on disk
            display: Show interactive view of plot

        Returns:
            None
        """
        test_image(ref_image)
        if not self.image_has_same_sampling(ref_image):
            self.resample(ref_image)

        fig, ax = plt.subplots(1, 1, figsize=[10, 8])

        scatter_mask = np.isfinite(ref_image.data)
        scatter_mask = np.where(np.isfinite(self.data), scatter_mask, False)
        ydata = self.data[scatter_mask]
        xdata = ref_image.data[scatter_mask]

        scatter_plot(
            ax,
            xdata,
            f"Reference image {ref_image.filename} [{ref_image.unit}]",
            ydata,
            f"{self.filename} [{self.unit}]",
            add_regression=True,
        )
        close_figure(
            fig,
            "Scatter plot against reference image",
            f"{destination}/{self.rootname}scatter.png",
            dpi,
            display,
        )


def image_comparison_chunk(compare_params):
    """
    Chunk function for parallel execution of the image comparison tool.
    Args:
        compare_params: Parameter dictionary for workflow control.

    Returns:
        A DataTree containing the Image and its reference Image.
    """

    image = FITSImage.from_fits_file(
        compare_params["this_image"], compare_params["telescope_name"]
    )
    ref_image = FITSImage.from_fits_file(
        compare_params["this_reference_image"], compare_params["telescope_name"]
    )
    plot_resampled = compare_params["plot_resampled"]
    plot_percentuals = compare_params["plot_percentuals"]
    plot_divided = compare_params["plot_divided_image"]
    plot_reference = compare_params["plot_reference"]
    plot_original = compare_params["plot_original"]
    destination = compare_params["destination"]
    colormap = compare_params["colormap"]
    dpi = compare_params["dpi"]
    display = compare_params["display"]

    if compare_params["comparison"] == "direct":
        image.compare_difference(ref_image)
        image.plot_images(
            destination,
            ref_image,
            plot_resampled,
            plot_percentuals,
            plot_reference,
            plot_original,
            False,
            colormap=colormap,
            dpi=dpi,
            display=display,
        )
    elif compare_params["comparison"] == "scaled":
        image.compare_scaled_difference(ref_image)
        image.plot_images(
            destination,
            ref_image,
            plot_resampled,
            plot_percentuals,
            plot_reference,
            plot_original,
            plot_divided,
            colormap=colormap,
            dpi=dpi,
            display=display,
        )
    else:
        raise Exception(f'Unknown comparison type {compare_params["comparison"]}')

    if compare_params["export_to_fits"]:
        image.export_to_fits(destination)

    if compare_params["plot_scatter"]:
        image.scatter_plot(destination, ref_image, dpi=dpi, display=display)

    img_node = xr.DataTree(name=image.filename, dataset=image.export_as_xds())
    ref_node = xr.DataTree(name=ref_image.filename, dataset=ref_image.export_as_xds())
    tree_node = xr.DataTree(
        name=image.rootname[:-1], children={"Reference": ref_node, "Image": img_node}
    )

    return tree_node


def extract_rms_from_xds(xds):
    """
    This simple function extracts FITSImage RMSes for a xds describing a FITSImage obj
    Args:
        xds: xds describing a FITSImage obj

    Returns:
        dict with RMS values
    """
    rms_dict = {}

    img_obj = FITSImage.from_xds(xds)
    if img_obj.residuals is None:
        rms_dict["resampled"] = np.nan
        rms_dict["residuals"] = np.nan
    else:
        rms_dict["resampled"] = np.nanstd(img_obj.mask_array(img_obj.data))
        rms_dict["residuals"] = np.nanstd(img_obj.mask_array(img_obj.residuals))

    rms_dict["original"] = np.nanstd(img_obj.mask_original())
    return rms_dict
