import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import xarray as xr
import pathlib

from astrohack.antenna.telescope import Telescope
from astrohack.utils.text import statistics_to_text
from astrohack.utils.algorithms import create_aperture_mask, data_statistics, are_axes_equal
from astrohack.visualization.plot_tools import well_positioned_colorbar, compute_extent
from astrohack.visualization.plot_tools import close_figure, get_proper_color_map, scatter_plot
from astrohack.utils.fits import read_fits, get_axis_from_fits_header, get_stokes_axis_iaxis, put_axis_in_fits_header, \
    write_fits


def test_image(fits_image):
    if isinstance(fits_image, FITSImage):
        pass
    else:
        raise TypeError('Reference image is not a FITSImage object')


class FITSImage:

    def __init__(self, filename: str, telescope_name: str):
        """
        Initializes the FITSImage object from a file on disk
        Args:
            filename: Name of the file on disk, may be .FITS file or a .zarr xds with a disk representation of a \
            FITSImage object
            telescope_name: Name of the telescope used on the images so that masking can be properly applied.
        """
        # Initialization from parameters
        self.filename = filename
        self.telescope_name = telescope_name
        self.rootname = '.'.join(filename.split('.')[:-1])+'.'

        # Blank slate initialization
        self.header = None
        self.data = None
        self.factor = 1.0
        self.residuals = None
        self.residuals_percent = None
        self.divided_image = None
        self.reference_name = None
        self.resampled = False
        self.x_axis = None
        self.y_axis = None
        self.original_x_axis = None
        self.original_y_axis = None
        self.x_unit = None
        self.y_unit = None
        self.unit = None
        self.fits_name = None
        self.original_data = None

        if '.FITS' in filename.upper():
            self._init_as_fits()
        elif '.zarr' in filename:
            self._init_as_xds()
        else:
            raise Exception(f"Don't know how to read {filename}")

    def _init_as_fits(self, istokes=0, ichan=0):
        self.header, self.data = read_fits(self.filename, header_as_dict=True)
        self.original_data = np.copy(self.data)
        self.fits_name = self.filename
        stokes_iaxis = get_stokes_axis_iaxis(self.header)

        self.unit = self.header['BUNIT']

        if len(self.data.shape) == 4:
            if stokes_iaxis == 4:
                self.data = self.data[istokes, ichan, ...]
            else:
                self.data = self.data[ichan, istokes, ...]

        elif len(self.data.shape) == 2:
            pass  # image is already as expected
        else:
            raise Exception(f'FITS image has an unsupported shape: {self.data.shape}')

        if 'AIPS' in self.header['ORIGIN']:
            self.x_axis, _, self.x_unit = get_axis_from_fits_header(self.header, 1, pixel_offset=False)
            self.y_axis, _, self.y_unit = get_axis_from_fits_header(self.header, 2, pixel_offset=False)
            self.x_unit = 'm'
            self.y_unit = 'm'
        elif 'Astrohack' in self.header['ORIGIN']:
            self.x_axis, _, self.x_unit = get_axis_from_fits_header(self.header, 1)
            self.y_axis, _, self.y_unit = get_axis_from_fits_header(self.header, 2)
            self.data = np.fliplr(self.data)
        else:
            raise Exception(f'Unrecognized origin:\n{self.header["origin"]}')
        self._create_base_mask()
        self.original_x_axis = np.copy(self.x_axis)
        self.original_y_axis = np.copy(self.y_axis)

    def _init_as_xds(self):
        filename = self.filename
        xds = xr.open_zarr(self.filename)
        for key in xds.attrs:
            setattr(self, key, xds.attrs[key])

        self.x_axis = xds.x.values
        self.y_axis = xds.y.values
        self.original_x_axis = xds.original_x.values
        self.original_y_axis = xds.original_y.values

        for key, value in xds.items():
            setattr(self, str(key), xds[key].values)

        self.filename = filename

    def _create_base_mask(self):
        telescope_obj = Telescope(self.telescope_name)
        self.base_mask = create_aperture_mask(self.x_axis, self.y_axis, telescope_obj.inlim, telescope_obj.oulim,
                                              arm_width=telescope_obj.arm_shadow_width,
                                              arm_angle=telescope_obj.arm_shadow_rotation)

    def resample(self, ref_image):
        test_image(ref_image)
        x_mesh_orig, y_mesh_orig = np.meshgrid(self.x_axis, self.y_axis, indexing='ij')
        x_mesh_dest, y_mesh_dest = np.meshgrid(ref_image.x_axis, ref_image.y_axis, indexing='ij')
        resamp = griddata((x_mesh_orig.ravel(), y_mesh_orig.ravel()), self.data.ravel(),
                          (x_mesh_dest.ravel(), y_mesh_dest.ravel()),
                          method='linear')
        size = ref_image.x_axis.shape[0], ref_image.y_axis.shape[0]
        self.x_axis = ref_image.x_axis
        self.y_axis = ref_image.y_axis
        self.data = resamp.reshape(size)
        self._create_base_mask()
        self.resampled = True

    def compare_difference(self, ref_image):
        test_image(ref_image)
        if not self.image_has_same_sampling(ref_image):
            self.resample(ref_image)

        self.residuals = ref_image.data - (self.data * self.factor)
        self.residuals_percent = 100 * self.residuals/ref_image.data
        self.reference_name = ref_image.filename

    def compare_scaled_difference(self, ref_image, rejection=10):
        test_image(ref_image)
        if not self.image_has_same_sampling(ref_image):
            self.resample(ref_image)
        simple_division = ref_image.data / self.data
        rough_factor = np.nanmean(simple_division[self.base_mask])
        self.divided_image = np.where(np.abs(simple_division) > rejection*rough_factor, np.nan, simple_division)
        self.factor = np.nanmedian(self.divided_image)
        self.compare_difference(ref_image)

    def image_has_same_sampling(self, ref_image):
        test_image(ref_image)
        return are_axes_equal(self.x_axis, ref_image.x_axis) and are_axes_equal(self.y_axis, ref_image.y_axis)

    def _mask_array(self, image_array):
        return np.where(self.base_mask, image_array, np.nan)

    def plot_images(self, destination, plot_data=False, plot_percentuals=False,
                    plot_divided_image=False, colormap='viridis', dpi=300, display=False):

        extent = compute_extent(self.x_axis, self.y_axis, 0.0)
        cmap = get_proper_color_map(colormap)
        base_name = f'{destination}/{self.rootname}'

        if self.residuals is None:
            raise Exception("Cannot plot results as they don't exist yet.")
        self._plot_map(self._mask_array(self.residuals), f'Residuals, ref={self.reference_name}',
                       f'Residuals [{self.unit}]', f'{base_name}residuals.png', cmap, extent,
                       'symmetrical', dpi, display, add_statistics=True)

        if plot_data:
            self._plot_map(self._mask_array(self.data), 'Original Data', f'Data [{self.unit}]',
                           f'{base_name}data.png', cmap, extent, [None, None], dpi, display,
                           add_statistics=False)

        if plot_percentuals:
            if self.residuals is None:
                raise Exception("Cannot plot results as they don't exist yet.")
            self._plot_map(self._mask_array(self.residuals_percent), f'Residuals in %, ref={self.reference_name}',
                           f'Residuals [%]', f'{base_name}residuals_percent.png', cmap, extent,
                           'symmetrical', dpi, display, add_statistics=True)

        if plot_divided_image:
            if self.divided_image is None:
                raise Exception("Cannot plot a divided image that does not exist.")
            self._plot_map(self._mask_array(self.divided_image), f'Divided image, ref={self.reference_name}',
                           f'Division [ ]', f'{base_name}divided.png', cmap, extent, [None, None],
                           dpi, display, add_statistics=True)

    def _plot_map(self, data, title, zlabel, filename, cmap, extent, zscale, dpi, display, add_statistics=False):
        fig, ax = plt.subplots(1, 1, figsize=[10, 8])
        if zscale == 'symmetrical':
            scale = max(np.abs(np.nanmin(data)), np.abs(np.nanmax(data)))
            vmin, vmax = -scale, scale
        else:
            vmin, vmax = zscale
            if vmin == 'None' or vmin is None:
                vmin = np.nanmin(data)
            if vmax == 'None' or vmax is None:
                vmax = np.nanmax(data)

        im = ax.imshow(data, cmap=cmap, interpolation="nearest", extent=extent,
                       vmin=vmin, vmax=vmax,)
        well_positioned_colorbar(ax, fig, im, zlabel, location='right', size='5%', pad=0.05)
        ax.set_xlabel(f"X axis [{self.x_unit}]")
        ax.set_ylabel(f"Y axis [{self.y_unit}]")
        if add_statistics:
            data_stats = data_statistics(data)
            ax.set_title(statistics_to_text(data_stats))
        close_figure(fig, title, filename, dpi, display)

    def export_as_xds(self):
        xds = xr.Dataset()
        obj_dict = vars(self)

        coords = {'x': self.x_axis, 'y': self.y_axis,
                  'original_x': self.original_x_axis, 'original_y': self.original_y_axis}
        for key, value in obj_dict.items():
            failed = False
            if isinstance(value, np.ndarray):
                if len(value.shape) == 2:
                    if 'original' in key:
                        xds[key] = xr.DataArray(value, dims=['original_x', 'original_y'])
                    else:
                        xds[key] = xr.DataArray(value, dims=['x', 'y'])
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
        xds = self.export_as_xds()
        xds.to_zarr(zarr_filename, mode="w", compute=True, consolidated=True)

    def __repr__(self):
        obj_dict = vars(self)
        outstr = ''
        for key, value in obj_dict.items():
            if isinstance(value, np.ndarray):
                outstr += f'{key:17s} -> {value.shape}'
            elif isinstance(value, dict):
                outstr += f'{key:17s} -> dict()'
            else:
                outstr += f'{key:17s} =  {value}'
            outstr += '\n'
        return outstr

    def export_to_fits(self, destination):
        pathlib.Path(destination).mkdir(exist_ok=True)
        ext_fits = '.fits'
        out_header = self.header.copy()

        put_axis_in_fits_header(out_header, self.x_axis, 1, '', self.x_unit)
        put_axis_in_fits_header(out_header, self.y_axis, 2, '', self.y_unit)

        obj_dict = vars(self)
        for key, value in obj_dict.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) == 2:
                    if 'original' in key:
                        pass
                    else:
                        if key == 'base_mask' or key == 'divided_image':
                            unit = ''

                        elif key == 'residuals_percent':
                            unit = '%'
                        else:
                            unit = self.unit
                        filename = f'{destination}/{self.rootname}{key}{ext_fits}'
                        write_fits(out_header, key, np.fliplr(value.astype(float)), filename, unit, reorder_axis=False)

    def scatter_plot(self, destination, ref_image, dpi=300, display=False):
        test_image(ref_image)
        if not self.image_has_same_sampling(ref_image):
            self.resample(ref_image)

        fig, ax = plt.subplots(1, 1, figsize=[10, 8])

        scatter_mask = np.isfinite(ref_image.data)
        scatter_mask = np.where(np.isfinite(self.data), scatter_mask, False)
        ydata = self.data[scatter_mask]
        xdata = ref_image.data[scatter_mask]

        scatter_plot(ax, xdata, f'Reference image {ref_image.filename} [{ref_image.unit}]',
                     ydata, f'{self.filename} [{self.unit}]', add_regression=True)
        close_figure(fig, 'Scatter plot against reference image', f'{destination}/{self.rootname}scatter.png',
                     dpi, display)


def image_comparison_chunk(compare_params):

    image = FITSImage(compare_params['this_image'], compare_params['telescope_name'])
    ref_image = FITSImage(compare_params['this_reference'], compare_params['telescope_name'])
    plot_data = compare_params['plot_data']
    plot_percentuals = compare_params['plot_percentuals']
    plot_divided = compare_params['plot_divided_image']
    destination = compare_params['destination']
    colormap = compare_params['colormap']
    dpi = compare_params['dpi']
    display = compare_params['display']

    if compare_params['comparison'] == 'direct':
        image.compare_difference(ref_image)
        image.plot_images(destination, plot_data, plot_percentuals, False, colormap=colormap, dpi=dpi,
                          display=display)
    elif compare_params['comparison'] == 'scaled':
        image.compare_scaled_difference(ref_image)
        image.plot_images(destination, plot_data, plot_percentuals, plot_divided, colormap=colormap, dpi=dpi,
                          display=display)
    else:
        raise Exception(f'Unknown comparison type {compare_params["comparison"]}')

    if compare_params['export_to_fits']:
        image.export_to_fits(destination)

    reference_node = xr.DataTree(name=ref_image.filename, data=ref_image.export_as_xds())
    tree_node = xr.DataTree(name=image.filename, data=image.export_as_xds(), children={'Reference': reference_node})

    return tree_node
