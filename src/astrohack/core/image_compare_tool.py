import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import xarray as xr

from astrohack.antenna.telescope import Telescope
from astrohack.utils.text import statistics_to_text
from astrohack.utils.algorithms import create_aperture_mask, data_statistics, are_axes_equal
from astrohack.visualization.plot_tools import well_positioned_colorbar, compute_extent
from astrohack.visualization.plot_tools import close_figure, get_proper_color_map
from astrohack.utils.fits import read_fits, get_axis_from_fits_header, get_stokes_axis_iaxis


def test_image(fits_image):
    if isinstance(fits_image, FITSImage):
        pass
    else:
        raise TypeError('Reference image is not a FITSImage object')


class FITSImage:

    def __init__(self, filename: str, telescope_name: str):
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
        self.x_unit = None
        self.y_unit = None
        self.unit = None
        self.fits_name = None

        if '.FITS' in filename.upper():
            self._init_as_fits(0, 0)
        elif '.zarr' in filename:
            self._init_as_xds()
        else:
            raise Exception(f"Don't know how to read {filename}")

    def _init_as_fits(self, istokes, ichan):
        self.header, self.data = read_fits(self.filename, header_as_dict=True)
        self.fits_name = self.filename
        stokes_iaxis = get_stokes_axis_iaxis(self.header)

        self.unit = self.header['BUNIT']

        if len(self.data.shape) == 4:
            if stokes_iaxis == 4:
                self.data = self.data[istokes, ichan, ...]
            else:
                self.data = self.data[ichan, istokes, ...]

        elif len(self.data.shape) == 2:
            pass # image is already as expected
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

    def _init_as_xds(self):
        xds = xr.open_zarr(self.filename)
        for key in xds.attrs:
            setattr(self, key, xds.attrs[key])

        self.x_axis = xds.x.values
        self.y_axis = xds.y.values

        for key, value in xds.items():
            setattr(self, key, xds[key].values)

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
        self.reference_name = ref_image.rootname

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

    def plot_images(self, destination, plot_residuals=True, plot_data=False, plot_percentuals=False,
                    plot_divided_image=False, colormap='viridis', dpi=300, display=False):

        extent = compute_extent(self.x_axis, self.y_axis, 0.0)
        cmap = get_proper_color_map(colormap)
        base_name = f'{destination}/{self.rootname}'

        if plot_residuals:
            if self.residuals is None:
                raise Exception("Cannot plot results as they don't exist yet.")
            self._plot_map(self._mask_array(self.residuals), 'Residuals', f'Residuals [{self.unit}]',
                           f'{base_name}residuals.png', cmap, extent, 'symmetrical', dpi, display,
                           add_statistics=True)

        if plot_data:
            self._plot_map(self._mask_array(self.data), 'Original Data', f'Data [{self.unit}]',
                           f'{base_name}data.png', cmap, extent, [None, None], dpi, display,
                           add_statistics=False)

        if plot_percentuals:
            if self.residuals is None:
                raise Exception("Cannot plot results as they don't exist yet.")
            self._plot_map(self._mask_array(self.residuals_percent), 'Residuals in %', f'Residuals [%]',
                           f'{base_name}residuals_percent.png', cmap, extent, 'symmetrical', dpi, display,
                           add_statistics=True)

        if plot_divided_image:
            if self.divided_image is None:
                raise Exception("Cannot plot a divided image that does not exist.")
            self._plot_map(self._mask_array(self.divided_image), 'Divided image', f'Division [ ]',
                           f'{base_name}divided.png', cmap, extent, [None, None], dpi, display,
                           add_statistics=True)

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

        coords = {'x': self.x_axis, 'y': self.y_axis}
        for key, value in obj_dict.items():
            failed = False
            if isinstance(value, np.ndarray):
                if len(value.shape) == 2:
                    xds[key] = xr.DataArray(value, dims=['x', 'y'])
                elif len(value.shape) == 1:
                    pass # Axes
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

    # def to_fits(self):
    #     fits = '.fits'
    #     header = self.header.copy()
    #     put_axis_in_header(self.x_axis, self.x_unit, 1, header)
    #     put_axis_in_header(self.y_axis, self.y_unit, 2, header)
    #
    #     if self.resampled:
    #         filename = f'comp_{self.rootname}.resampled'
    #     else:
    #         filename = f'comp_{self.rootname}'
    #
    #     create_fits(header, self.masked, f'{filename}.masked{fits}')
    #     create_fits(header, self.mask, f'{filename}.mask{fits}')
    #
    #     if self.division is not None:
    #         create_fits(header, self.division, f'{filename}.division{fits}')
    #
    #     if self.residuals is not None:
    #         create_fits(header, self.residuals, f'{filename}.residual{fits}')
    #
    #
    #     if self.noise is not None:
    #         create_fits(header, self.noise, f'{filename}.noise{fits}')


    # def print_stats(self):
    #     print(80*'*')
    #     print()
    #     print(f'Mean scaling factor = {self.factor:.3}')
    #     print(f'Mean Residual = {self.res_mean:.3}%')
    #     print(f'Residuals RMS = {self.res_rms:.3}%')
    #     print()


# instatiation
# first_image = image(args.first, args.noise_clip, args.blocage,
#                    args.diameter/2, args.no_division, args.shadow_width,
#                    args.shadow_rotation)
# second_image = image(args.second, args.noise_clip, args.blocage,
#                         args.diameter/2, args.no_division, args.shadow_width,
#                         args.shadow_rotation)
#
# # Data manipulation
# second_image.resample(first_image)
# first_image.make_comparison(second_image)
#
# # Plotting
# first_image.plot(args.noise_map, args.colormap, args.first_zscale)
# second_image.plot(args.noise_map, args.colormap, args.second_zscale)
#
# if args.fits:
#     first_image.to_fits()
#     second_image.to_fits()
#
# if not args.quiet:
#     first_image.print_stats()





