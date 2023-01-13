import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astrohack._classes.linear_axis import LinearAxis
from astrohack._classes.ring_panel import RingPanel
from astrohack._utils._fits_io import _read_fits
from astrohack._utils._fits_io import _write_fits

lnbr = '\n'

def _convert_to_db(val: float):
    """
    Converts a float value to decibels
    Args:
        val (float): Value to be converted to decibels
    Returns:
        Value in decibels
    """
    return 10. * np.log10(val)

class AntennaSurface:

    def __init__(self, amp, dev, telescope, cutoff=0.21, pkind=None):
        """
        Antenna Surface description capable of computing RMS, Gains, and fitting the surface to obtain screw adjustments
        Args:
            amp: Amplitude aperture image (AIPS FITS file)
            dev: Physical deviation aperture image (AIPS FITS file)
            telescope: currently supported: ["VLA","VLBA"]
            cutoff: fractional cutoff on the amplitude image to exclude regions with weak amplitude from the panel
            surface fitting
            pkind: Kind of panel surface fitting, if is None defaults to telescope default
        """
        # Initializes antenna surface parameters
        self.ampfile = amp
        self.devfile = dev

        self.ingains = np.nan
        self.ougains = np.nan
        self.inrms = np.nan
        self.ourms = np.nan

        self._read_images()
        self.cut = cutoff * np.max(self.amp)

        if telescope == 'VLA':
            self._init_vla()
        elif telescope == 'VLBA':
            self._init_vlba()
        else:
            raise Exception("Unknown telescope: " + telescope)
        if pkind is None:
            pass
        else:
            self.panelkind = pkind

        self._get_aips_headpars()
        self.reso = self.diam / self.npoint

        self.resi = None
        self.corr = None
        self.solved = False
        if self.ringed:
            self._build_polar()
            self._build_ring_panels()
            self._build_ring_mask()
            self.fetch_panel = self._fetch_panel_ringed
            self.compile_panel_points = self._compile_panel_points_ringed

    def _get_aips_headpars(self):
        """
        Fetches AIPS specific metadata from FITS headers
        """
        for line in self.devhead["HISTORY"]:
            wrds = line.split()
            if wrds[1] == "Visibilities":
                self.npoint = np.sqrt(int(wrds[-1]))
            elif wrds[1] == "Observing":
                # Stored in mm
                self.wavel = 1000 * float(wrds[-2])
            elif wrds[1] == "Antenna" and wrds[2] == "surface":
                self.inlim = abs(float(wrds[-3]))
                self.oulim = abs(float(wrds[-2]))

    def _read_images(self):
        """
        Reads amplitude and deviation images and initializes the X and Y axes
        """
        self.amphead, self.amp = _read_fits(self.ampfile)
        self.devhead, self.dev = _read_fits(self.devfile)
        self.dev *= 1000
        #
        if self.devhead['NAXIS1'] != self.amphead['NAXIS1']:
            raise Exception("Amplitude and deviation images have different sizes")
        self.npix = int(self.devhead['NAXIS1'])
        self.xaxis = LinearAxis(self.npix, self.amphead["CRPIX1"],
                                self.amphead["CRVAL1"], self.amphead["CDELT1"])
        self.yaxis = LinearAxis(self.npix, self.amphead["CRPIX2"],
                                self.amphead["CRVAL2"], self.amphead["CDELT2"])
        return

    def _build_ring_mask(self):
        """
        Builds the mask on regions to be included in panel surface masks, specific to circular antennas as there is an
        outer and inner limit to the mask based on the antenna's inner receiver hole and outer edge
        """
        self.mask = np.where(self.amp < self.cut, False, True)
        self.mask = np.where(self.rad > self.inlim, self.mask, False)
        self.mask = np.where(self.rad < self.oulim, self.mask, False)
        self.mask = np.where(np.isnan(self.dev), False, self.mask)

    # Other known telescopes should be included here, ALMA, ngVLA
    def _init_vla(self):
        """
        Initializes object according to parameters specific to VLA panel distribution
        """
        self.panelkind = 'flexible'
        self.telescope = "VLA"
        self.diam = 25.0  # meters
        self.focus = 8.8  # meters
        self.ringed = True
        self.nrings = 6
        self.npanel = [12, 16, 24, 40, 40, 40]
        self.inrad = [1.983, 3.683, 5.563, 7.391, 9.144, 10.87]
        self.ourad = [3.683, 5.563, 7.391, 9.144, 10.87, 12.5]
        self.inlim = 2.0
        self.oulim = 12.0

    def _init_vlba(self):
        """
        Initializes object according to parameters specific to VLBA panel distribution
        """
        self.panelkind = "flexible"
        self.telescope = "VLBA"
        self.diam = 25.0  # meters
        self.focus = 8.75  # meters
        self.ringed = True
        self.nrings = 6
        self.npanel = [20, 20, 40, 40, 40, 40]
        self.inrad = [1.676, 3.518, 5.423, 7.277, 9.081, 10.808]
        self.ourad = [3.518, 5.423, 7.277, 9.081, 10.808, 12.500]
        self.inlim = 2.0
        self.oulim = 12.0

    def _build_polar(self):
        """
        Build polar coordinate grid, specific for circular antennas with panels arranged in rings
        """
        self.rad = np.zeros([self.npix, self.npix])
        self.phi = np.zeros([self.npix, self.npix])
        for iy in range(self.npix):
            ycoor = self.yaxis.idx_to_coor(iy + 0.5)
            for ix in range(self.npix):
                xcoor = self.xaxis.idx_to_coor(ix + 0.5)
                self.rad[ix, iy] = np.sqrt(xcoor ** 2 + ycoor ** 2)
                self.phi[ix, iy] = np.arctan2(ycoor, xcoor)
                if self.phi[ix, iy] < 0:
                    self.phi[ix, iy] += 2 * np.pi

    def _build_ring_panels(self):
        """
        Build list of panels, specific for circular antennas with panels arranged in rings
        """
        self.panels = []
        for iring in range(self.nrings):
            angle = 2.0 * np.pi / self.npanel[iring]
            for ipanel in range(self.npanel[iring]):
                panel = RingPanel(self.panelkind, angle, iring,
                                  ipanel, self.inrad[iring], self.ourad[iring])
                self.panels.append(panel)
        return

    def _compile_panel_points_ringed(self):
        """
        Loops through the points in the antenna surface and checks to which panels it belongs,
        specific for circular antennas with panels arranged in rings
        """
        for iy in range(self.npix):
            yc = self.yaxis.idx_to_coor(iy + 0.5)
            for ix in range(self.npix):
                if self.mask[ix, iy]:
                    xc = self.xaxis.idx_to_coor(ix + 0.5)
                    # How to do the coordinate choice here without
                    # adding an if?
                    for panel in self.panels:
                        if panel.is_inside(self.rad[ix, iy], self.phi[ix, iy]):
                            panel.add_point([xc, yc, ix, iy, self.dev[ix, iy]])

    def _fetch_panel_ringed(self, ring, panel):
        """
        Fetch a panel object from the panel list using its ring and panel numbers,
        specific for circular antennas with panels arranged in rings
        Args:
            ring: Ring number
            panel: Panel number

        Returns:
        Panel object
        """
        if ring == 1:
            ipanel = panel - 1
        else:
            ipanel = np.sum(self.npanel[:ring - 1]) + panel - 1
        return self.panels[ipanel]

    def gains(self):
        """
        Computes antenna gains in decibels before and after panel surface fitting
        Returns:
        Gains before panel fitting OR Gains before and after panel fitting
        """
        self.ingains = self._gains_array(self.dev)
        if self.resi is None:
            return self.ingains
        else:
            self.ougains = self._gains_array(self.resi)
            return self.ingains, self.ougains

    def _gains_array(self, arr):
        """
        Worker for gains method, works with the actual arrays to compute the gains
        Args:
            arr: Deviation image over which to compute the gains

        Returns:
        Actual and theoretical gains
        """
        # Compute the actual and theoretical gains for the current
        # antenna surface. What is the unit for the wavelength? mm
        forpi = 4.0 * np.pi
        fact = 1000. * self.reso / self.wavel
        fact *= fact
        #
        # What are these sums?
        sumrad = 0.0
        sumtheta = 0.0
        nsamp = 0
        #    convert surface error to phase
        #    and compute gain loss
        for iy in range(self.npix):
            for ix in range(self.npix):
                if self.mask[ix, iy]:
                    quo = self.rad[ix, iy] / (2. * self.focus)
                    phase = arr[ix, iy] * forpi / (np.sqrt(1. + quo * quo) * self.wavel)
                    sumrad += np.cos(phase)
                    sumtheta += np.sin(phase)
                    nsamp += 1

        ampmax = np.sqrt(sumrad * sumrad + sumtheta * sumtheta)
        if nsamp <= 0:
            raise Exception("Antenna is blanked")
        ampmax *= fact / nsamp
        gain = ampmax * forpi
        thgain = fact * forpi
        #
        gain = _convert_to_db(gain)
        thgain = _convert_to_db(thgain)
        return gain, thgain

    def get_rms(self):
        """
        Computes antenna surface RMS before and after panel surface fitting
        Returns:
        RMS before panel fitting OR RMS before and after panel fitting
        """
        self.inrms = np.sqrt(np.mean(self.dev[self.mask] ** 2))
        if self.resi is None:
            return self.inrms
        else:
            self.ourms = np.sqrt(np.mean(self.resi[self.mask] ** 2))
            return self.inrms, self.ourms

    def fit_surface(self):
        """
        Loops over the panels to fit the panel surfaces
        """
        for panel in self.panels:
            panel.solve()
        self.solved = True

    def correct_surface(self):
        """
        Apply corrections determined by the panel surface fitting methods to the antenna surface
        """
        if not self.solved:
            raise Exception("Panels must be fitted before atempting a correction")
        self.corr = np.where(self.mask, 0, np.nan)
        self.resi = np.copy(self.dev)
        for panel in self.panels:
            panel.get_corrections()
            for ipnt in range(len(panel.corr)):
                val = panel.values[ipnt]
                ix, iy = int(val[2]), int(val[3])
                self.resi[ix, iy] -= panel.corr[ipnt]
                self.corr[ix, iy] = -panel.corr[ipnt]

    def print_misc(self):
        """
        Print miscelaneous information on the panels in the antenna surface
        """
        for panel in self.panels:
            panel.print_misc()

    def plot_surface(self, filename=None, mask=False, screws=False, dpi=300):
        """
        Do plots of the antenna surface
        Args:
            filename: Save plot to a file rather than displaying it with matplotlib widgets
            mask: Display mask and amplitudes rather than deviation images
            screws: Display the screws on the panels
        """
        vmin, vmax = np.nanmin(self.dev), np.nanmax(self.dev)
        rms = self.get_rms()
        if mask:
            fig, ax = plt.subplots(1, 2, figsize=[10, 5])
            title = "Mask"
            self._plot_surface(self.mask, title, fig, ax[0], 0, 1, screws=screws,
                               mask=mask)
            vmin, vmax = np.nanmin(self.amp), np.nanmax(self.amp)
            title = "Amplitude min={0:.5f}, max ={1:.5f}".format(vmin, vmax)
            self._plot_surface(self.amp, title, fig, ax[1], vmin, vmax, screws=screws,
                               unit=self.amphead["BUNIT"].strip())
        else:
            if self.resi is None:
                fig, ax = plt.subplots()
                title = "Before correction\nRMS = {0:8.5} mm".format(rms)
                self._plot_surface(self.dev, title, fig, ax, vmin, vmax, screws=screws)
            else:
                fig, ax = plt.subplots(1, 3, figsize=[15, 5])
                title = "Before correction\nRMS = {0:.3} mm".format(rms[0])
                self._plot_surface(self.dev, title, fig, ax[0], vmin, vmax, screws=screws)
                title = "Corrections"
                self._plot_surface(self.corr, title, fig, ax[1], vmin, vmax, screws=screws)
                title = "After correction\nRMS = {0:.3} mm".format(rms[1])
                self._plot_surface(self.resi, title, fig, ax[2], vmin, vmax, screws=screws)
        fig.suptitle("Antenna Surface")
        fig.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, dpi=dpi)

    def _plot_surface(self, data, title, fig, ax, vmin, vmax, screws=False, mask=False,
                      unit='mm'):
        """
        Does the plotting of a data array in a figure's subplot
        Args:
            data: The array to be plotted
            title: Title of the subplot
            fig: Global figure containing the subplots
            ax: matplotlib axes instance describing the subplot
            vmin: minimum to the color scale
            vmax: maximum to the color scale
            screws: Display screws
            mask: do not add colorbar if plotting a mask
            unit: Unit of the data in the color scale
        """
        ax.set_title(title)
        # set the limits of the plot to the limits of the data
        xmin = self.xaxis.idx_to_coor(-0.5)
        xmax = self.xaxis.idx_to_coor(self.xaxis.n - 0.5)
        ymin = self.yaxis.idx_to_coor(-0.5)
        ymax = self.yaxis.idx_to_coor(self.yaxis.n - 0.5)
        im = ax.imshow(np.flipud(data), cmap='viridis', interpolation='nearest',
                       extent=[xmin, xmax, ymin, ymax], vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        if not mask:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, label="Z Scale [" + unit + "]", cax=cax)
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")
        for panel in self.panels:
            panel.plot(ax, screws=screws)

    def export_corrected(self, filename):
        """
        Export corrected surface to a FITS file
        Args:
            filename: Output FITS file name/path
        """
        if self.resi is None:
            raise Exception("Cannot export corrected surface")
        _write_fits(self.devhead,self.resi,filename)
        return

    def export_screw_adjustments(self, filename, unit='mm'):
        """
        Export screw adjustments for all panels onto an ASCII file
        Args:
            filename: ASCII file name/path
            unit: unit for panel screw adjustments ['mm','miliinches']
        """
        spc = ' '
        outfile = 'Screw adjustments for {0:s} {1:s} antenna\n'.format(
            self.telescope, self.amphead['telescop'])
        outfile += 'Adjustments are in ' + unit + lnbr
        outfile += 2 * lnbr
        outfile += 25 * spc + "{0:22s}{1:22s}".format('Inner Edge', 'Outer Edge') + lnbr
        outfile += 5 * spc + "{0:8s}{1:8s}".format("Ring", "panel")
        outfile += 2 * spc + 2 * "{0:11s}{1:11s}".format('left', 'right') + lnbr
        for panel in self.panels:
            outfile += panel.export_adjustments(unit=unit) + lnbr
        lefile = open(filename, 'w')
        lefile.write(outfile)
        lefile.close()
