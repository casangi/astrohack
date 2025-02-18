from astropy.io import fits
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from astrohack.visualization.plot_tools import well_positioned_colorbar
from astrohack.visualization.plot_tools import close_figure, get_proper_color_map
import datetime

parser = argparse.ArgumentParser(description="Compare aperture FTIS maps produced by AIPS and AstroHACK\n"
                                 "Beam maps are not supported")
parser.add_argument('first', type=str, help='first aperture FITS (usually AIPS)')
parser.add_argument('second', type=str,
                    help='Second aperture FITS (usually AstroHACK)')
parser.add_argument('-n', '--noise-map', action='store_true',
                    default=False, help='Save noise images')
parser.add_argument('-c', '--noise-clip', type=float, default=1,
                    help='Noise clipping level (in sigmas)')
parser.add_argument('-q', '--quiet', action='store_true', default=False,
                    help='Do not print value to screen')
parser.add_argument('-d', '--diameter', type=float, default=25.0,
                    help='Dish diameter')
parser.add_argument('-b', '--blocage', type=float, default=2.0,
                    help='Dish inner blocage radius')
parser.add_argument('-m', '--colormap', type=str, default='viridis',
                    help='Colormap for non residual maps')
parser.add_argument('-p', '--no-division', action='store_true', default=False,
                    help='Do not perform division, i.e. factor assumed to be 1')
parser.add_argument('-s', '--shadow-width', type=float, default=1.5,
                    help='Arm shadow width in meters')
parser.add_argument('-r', '--shadow-rotation', type=float, default=0,
                     help='Arm shadow rotation in degrees,(e.g. 45 degress for some ALMA antennas)')
parser.add_argument('-z', '--first-zscale', type=float, nargs=2, default=[None, None],
                    help='Z scale for first image (min max)')
parser.add_argument('-y', '--second-zscale', type=float, nargs=2, default=[None, None],
                    help='Z scale for second image (min max)')
parser.add_argument('-f', '--fits', action='store_true',
                    default=False, help='Save products as FITS images')

args = parser.parse_args()

def get_axis_from_header(header, iaxis):
    n_elem = header[f'NAXIS{iaxis}']
    ref = header[f'CRPIX{iaxis}']
    val = header[f'CRVAL{iaxis}']
    inc = header[f'CDELT{iaxis}']
    axis = np.ndarray((n_elem))
    for i_elem in range(n_elem):
        axis[i_elem] = val+(ref-i_elem)*inc
    return axis


def put_axis_in_header(axis, unit, iaxis, header):
    n_elem = len(axis)
    ref = n_elem//2
    val = axis[ref]
    inc = axis[1]-axis[0]
    header[f'NAXIS{iaxis}'] = n_elem
    header[f'CRPIX{iaxis}'] = ref
    header[f'CRVAL{iaxis}'] = val
    header[f'CDELT{iaxis}'] = inc
    header[f'CUNIT{iaxis}'] = unit


def create_fits(header, data, filename):
    hdu = fits.PrimaryHDU(data)
    for key, value in header.items():
        if isinstance(value, str):
            if '/' in value:
                wrds = value.split('/')
                hdu.header.set(key, wrds[0], wrds[1])
        else:
            hdu.header.set(key, value)
    hdu.header.set('ORIGIN', f'Image comparison code')
    hdu.header.set('DATE', datetime.datetime.now().strftime('%b %d %Y, %H:%M:%S'))
    hdu.writeto(filename, overwrite=True)


def vers_comp_recursive(ref, cur):
    if ref[0] > cur[0]:
        return -1
    elif ref[0] < cur[0]:
        return 1
    else:
        if len(ref) == 1:
            return 0
        else:
            return vers_comp_recursive(ref[1:], cur[1:])


def test_version(reference, current):
    ref_num = get_numbers_from_version(reference)
    cur_num = get_numbers_from_version(current)
    return vers_comp_recursive(ref_num, cur_num)


def get_numbers_from_version(version):
    numbers = version[1:].split('.')
    revision = int(numbers[0])
    major = int(numbers[1])
    minor = int(numbers[2])
    return [revision, major,  minor]


class image:

    def __init__(self, filename, noise_level, inlim, oulim, no_division, arm_width, arm_angle):
        self.no_division = no_division
        self.inlim = inlim
        self.oulim = oulim
        self.noise_level = noise_level
        self.arm_width = arm_width
        self.arm_angle = arm_angle*np.pi/180

        opn_fits =  fits.open(filename)
        self.data = opn_fits[0].data[0,0,:,:]
        self._get_info_from_header(opn_fits[0].header)
        opn_fits.close()

        self._mask_image()

        self.division = None
        self.residuals = None
        self.resampled = False
        self.res_mean = None
        self.res_rms = None
        self.rootname = '.'.join(filename.split('.')[:-1])


    def _astrohack_specific_init(self, header):
        self.x_unit = header['CUNIT1']
        self.y_unit = header['CUNIT2']
        version = header['ORIGIN'].split()[1][:-1]
        if test_version('v0.4.1', version) <= 0:
            # This treatment is only necessary before v0.4.2
            wavelength = header['WAVELENG']
            self.x_axis *= wavelength
            self.y_axis *= wavelength
            self.data = np.fliplr(np.flipud(self.data))
        if test_version('v0.5.0', version) >= 0:
            print('estoy aqui?')
            wavelength = header['WAVELENG']
            self.x_axis /= wavelength
            self.y_axis /= wavelength
            self.data = np.fliplr(np.flipud(self.data))


    def _get_info_from_header(self, header):
        self.header = header
        self.x_axis = get_axis_from_header(header, 1)
        self.y_axis = get_axis_from_header(header, 2)
        self.unit = header['BUNIT']

        if 'AIPS' in header['ORIGIN']:
            self.x_unit = 'm'
            self.y_unit = 'm'
        elif 'Astrohack' in header['ORIGIN']:
            self._astrohack_specific_init(header)
        else:
            raise Exception(f'Unrecognized origin:\n{header["origin"]}')


    def _mask_image(self):
        x_mesh, y_mesh = np.meshgrid(self.x_axis, self.y_axis)
        self.radius = np.sqrt(x_mesh**2 + y_mesh**2)
        mask = np.where(self.radius > self.oulim, np.nan, 1.0)
        mask = np.where(self.radius < self.inlim, np.nan, mask)

        # Arm masking
        if self.arm_angle%np.pi == 0:
            mask = np.where(np.abs(x_mesh) < self.arm_width/2., np.nan, mask)
            mask = np.where(np.abs(y_mesh) < self.arm_width/2., np.nan, mask)
        else:
            # first shadow
            coeff = np.tan(self.arm_angle%np.pi)
            distance = np.abs((coeff*x_mesh-y_mesh)/np.sqrt(coeff**2+1))
            mask = np.where(distance < self.arm_width/2., np.nan, mask)
            # second shadow
            coeff = np.tan(self.arm_angle%np.pi+np.pi/2)
            distance = np.abs((coeff*x_mesh-y_mesh)/np.sqrt(coeff**2+1))
            mask = np.where(distance < self.arm_width/2., np.nan, mask)


        if self.no_division:
            self.noise = None
            self.rms = None
        else:
            self.noise, self.rms = self._noise_filter()
            mask = np.where(self.data<self.noise_level*self.rms, np.nan, mask)

        self.mask = mask
        self.masked = self.data*mask


    def resample(self, ref_image):
        x_mesh_orig, y_mesh_orig = np.meshgrid(self.x_axis, self.y_axis)
        x_mesh_dest, y_mesh_dest = np.meshgrid(ref_image.x_axis, ref_image.y_axis)
        resamp = griddata((x_mesh_orig.ravel(), y_mesh_orig.ravel()),
                          self.data.ravel(),
                          (x_mesh_dest.ravel(), y_mesh_dest.ravel()),
                          method='linear')
        size = ref_image.x_axis.shape[0], ref_image.y_axis.shape[0]
        self.x_axis = ref_image.x_axis
        self.y_axis = ref_image.y_axis
        self.data = resamp.reshape(size)
        self._mask_image()
        self.resampled = True


    def _noise_filter(self):
        noise = np.where(self.radius < self.oulim, np.nan, self.data)
        noise = np.where(self.radius < self.inlim, self.data, noise)
        noiserms = np.sqrt(np.nanmean(noise**2))
        return noise, noiserms


    def make_comparison(self, image2):
        if self.no_division:
            self.factor = 1.0
            self.division = None
        else:
            self._compute_divide(image2)

        self._compute_residuals(image2)


    def _compute_divide(self, image2):
        division = self.masked/image2.masked
        rude_factor = np.abs(np.nanmean(division))
        self.division = np.where(np.abs(division)>10*rude_factor, np.nan, division)
        self.factor = np.nanmedian(self.division)


    def _compute_residuals(self, image2, blanking=100):
         percent = 100*(self.masked-self.factor*image2.masked)/self.masked
         percent = np.where(np.abs(percent)>blanking, np.nan, percent)
         self.residuals = percent
         self.res_mean = np.nanmean(percent)
         self.res_rms = np.sqrt(np.nanmean(percent**2))


    def _plot_map(self, data, title, label, filename, cmap, extent, zscale):
        fig, ax = plt.subplots(1, 1, figsize=[10,8])
        cmap = get_proper_color_map(cmap)
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
        well_positioned_colorbar(ax, fig, im, label, location='right', size='5%', pad=0.05)
        ax.set_xlabel(f"X axis [{self.x_unit}]")
        ax.set_ylabel(f"Y axis [{self.y_unit}]")
        close_figure(fig, title, filename, 300, False)


    def plot(self, plot_noise, cmap, z_scale):
        extent = [np.min(self.x_axis), np.max(self.x_axis), np.min(self.y_axis), np.max(self.y_axis)]
        png = '.png'
        if self.resampled:
            title = f'Resampled {self.rootname}'
            filename = f'{self.rootname}.resampled'
        else:
            title = f'{self.rootname}'
            filename = f'{self.rootname}'

        if self.no_division:
            zlabel = f'?What is type? [{self.unit}]'
        else:
            zlabel = f'Amplitude [{self.unit}]'

        self._plot_map(self.masked, title, zlabel, filename+png, cmap, extent, z_scale)
        self._plot_map(self.mask, f'Mask used for {self.rootname}', 'Mask value',
                       f'{self.rootname}.mask{png}', cmap, extent, [0, 1])

        if self.division is not None:
            self._plot_map(self.division, f'Division map, mean factor:{self.factor:.3}',
                           'Divided value [ ]',
                           f'{filename}.division{png}', cmap, extent, [None, None])

        if self.residuals is not None:
            self._plot_map(self.residuals,
                           f'Residual map, mean residual: {self.res_mean:.3}%, residual RMS: '
                           f'{self.res_rms:.3}%',
                           'Residuals [%]',
                           f'{filename}.residual{png}', 'RdBu_r', extent, 'symmetrical')

        if plot_noise:
            if self.noise is not None:
                self._plot_map(self.noise,
                               f'Noise like component for {self.rootname}', zlabel,
                               f'{filename}.noise{png}', cmap, extent, [None, None])


    def to_fits(self):
        fits = '.fits'
        header = self.header.copy()
        put_axis_in_header(self.x_axis, self.x_unit, 1, header)
        put_axis_in_header(self.y_axis, self.y_unit, 2, header)

        if self.resampled:
            filename = f'comp_{self.rootname}.resampled'
        else:
            filename = f'comp_{self.rootname}'

        create_fits(header, self.masked, f'{filename}.masked{fits}')
        create_fits(header, self.mask, f'{filename}.mask{fits}')

        if self.division is not None:
            create_fits(header, self.division, f'{filename}.division{fits}')

        if self.residuals is not None:
            create_fits(header, self.residuals, f'{filename}.residual{fits}')


        if self.noise is not None:
            create_fits(header, self.noise, f'{filename}.noise{fits}')


    def print_stats(self):
        print(80*'*')
        print()
        print(f'Mean scaling factor = {self.factor:.3}')
        print(f'Mean Residual = {self.res_mean:.3}%')
        print(f'Residuals RMS = {self.res_rms:.3}%')
        print()


# instatiation
first_image = image(args.first, args.noise_clip, args.blocage,
                   args.diameter/2, args.no_division, args.shadow_width,
                   args.shadow_rotation)
second_image = image(args.second, args.noise_clip, args.blocage,
                        args.diameter/2, args.no_division, args.shadow_width,
                        args.shadow_rotation)

# Data manipulation
second_image.resample(first_image)
first_image.make_comparison(second_image)

# Plotting
first_image.plot(args.noise_map, args.colormap, args.first_zscale)
second_image.plot(args.noise_map, args.colormap, args.second_zscale)

if args.fits:
    first_image.to_fits()
    second_image.to_fits()

if not args.quiet:
    first_image.print_stats()





