import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps

from astrohack._classes.base_panel import panel_models, irigid
from astrohack._classes.ring_panel import RingPanel
from astrohack._utils._constants import *
from astrohack._utils._conversion import _convert_to_db
from astrohack._utils._conversion import _convert_unit
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._tools import _add_prefix, _well_positioned_colorbar, _axis_to_fits_header
from astrohack._utils._io import _write_fits

lnbr = "\n"
figsize = [5, 4]


class AntennaSurface:
    def __init__(self, inputxds, telescope, cutoff=None, pmodel=None, crop=False, panel_margins=None, reread=False):
        """
        Antenna Surface description capable of computing RMS, Gains, and fitting the surface to obtain screw adjustments
        Args:
            inputxds: Input xarray dataset
            telescope: Telescope object
            cutoff: fractional cutoff on the amplitude image to exclude regions with weak amplitude from the panel,
                    defaults to 20% if None
            pmodel: model of panel surface fitting, if is None defaults to telescope default
            crop: Crop apertures to slightly larger frames than the antenna diameter
            panel_margins: Margin to be ignored at edges of panels when fitting, defaults to 20% if None
            reread: Read a previously processed holography
        """
        self.reread = reread
        self._nullify()
        self._read_xds(inputxds)
        self.telescope = telescope

        if not self.reread:
            if cutoff is None:
                self.cut = 0.2 * np.nanmax(self.amplitude)
            else:
                self.cut = cutoff * np.nanmax(self.amplitude)
            if pmodel is None:
                self.panelmodel = panel_models[irigid]
            else:
                self.panelmodel = pmodel
            if panel_margins is None:
                self.panel_margins = 0.2
            else:
                self.panel_margins = panel_margins
            self.reso = self.telescope.diam / self.npoint

            if crop:
                self._crop_maps()
        if self.telescope.ringed:
            self._init_ringed()
        if not self.reread:
            if self.computephase:
                self.phase = self._deviation_to_phase(self.deviation)
            else:
                self.deviation = self._phase_to_deviation(self.phase)

            self.phase = self._nan_out_of_bounds(self.phase)
            self.amplitude = self._nan_out_of_bounds(self.amplitude)
            self.deviation = self._nan_out_of_bounds(self.deviation)

    def _read_aips_xds(self, inputxds):
        self.amplitude = np.flipud(inputxds["AMPLITUDE"].values)
        self.deviation = np.flipud(inputxds["DEVIATION"].values)
        self.npoint = inputxds.attrs['npoint']
        self.wavelength = inputxds.attrs['wavelength']
        self.amp_unit = inputxds.attrs['amp_unit']
        self.u_axis = inputxds.u.values
        self.v_axis = inputxds.v.values
        self.computephase = True
        self.processed = False

    def _read_holog_xds(self, inputxds):
        if 'chan' in inputxds.dims:
            if inputxds.dims['chan'] != 1:
                raise Exception("Only single channel holographies supported")
            self.wavelength = clight / inputxds.chan.values[0]
        else:
            self.wavelength = inputxds.attrs['wavelength']

        self.amplitude = inputxds["AMPLITUDE"].values[0, 0, 0, :, :]
        self.phase = inputxds["CORRECTED_PHASE"].values[0, 0, 0, :, :]

        self.npoint = np.sqrt(inputxds.dims['l'] ** 2 + inputxds.dims['m'] ** 2)
        self.amp_unit = 'V'
        self.u_axis = inputxds.u_prime.values * self.wavelength
        self.v_axis = inputxds.v_prime.values * self.wavelength
        self.computephase = False

    def _read_panel_xds(self, inputxds):
        self.wavelength = inputxds.attrs['wavelength']
        self.amp_unit = inputxds.attrs['amp_unit']
        self.panelmodel = inputxds.attrs['panel_model']
        self.panel_margins = inputxds.attrs['panel_margin']
        self.cut = inputxds.attrs['cutoff']
        self.solved = inputxds.attrs['solved']
        self.fitted = inputxds.attrs['fitted']
        # Arrays
        self.amplitude = inputxds['AMPLITUDE'].values
        self.phase = inputxds['PHASE'].values
        self.deviation = inputxds['DEVIATION'].values
        self.mask = inputxds['MASK']
        self.u_axis = inputxds.u.values
        self.v_axis = inputxds.u.values
        self.panel_distribution = inputxds['PANEL_DISTRIBUTION'].values

        if self.solved:
            self.phase_residuals = inputxds['PHASE_RESIDUALS'].values
            self.residuals = inputxds['RESIDUALS'].values
            self.phase_corrections = inputxds['PHASE_CORRECTIONS'].values
            self.corrections = inputxds['CORRECTIONS'].values
            self.panel_pars = inputxds['PANEL_PARAMETERS'].values
            self.screw_adjustments = inputxds['PANEL_SCREWS'].values
            self.panel_labels = inputxds.labels.values

    def _read_xds(self, inputxds):
        """
        Read input XDS, distinguishing what is derived from AIPS data and what was created by astrohack.holog
        Args:
            inputxds: X array dataset
        """
        # Origin dependant reading
        if self.reread:
            self._read_panel_xds(inputxds)
        else:
            if inputxds.attrs['AIPS']:
                self._read_aips_xds(inputxds)
            else:
                self._read_holog_xds(inputxds)

        # Common elements
        self.unpix = self.u_axis.shape[0]
        self.vnpix = self.v_axis.shape[0]
        self.antenna_name = inputxds.attrs['ant_name']
        self.ddi = inputxds.attrs['ddi']

    def _nullify(self):
        """
        Part of the initialization process, nullify the data objects to be used later
        """
        self.phase = None
        self.deviation = None
        self.residuals = None
        self.corrections = None
        self.phase_corrections = None
        self.phase_residuals = None
        self.solved = False
        self.ingains = np.nan
        self.ougains = np.nan
        self.in_rms = np.nan
        self.out_rms = np.nan
        self.fitted = False

    def _init_ringed(self):
        """
        Do the proper method association for the case of a ringed antenna
        """
        if self.telescope.panel_numbering == 'ring, clockwise, top':
            self._panel_label = self._vla_panel_labeling
        elif self.telescope.panel_numbering == 'sector, counterclockwise, right':
            self._panel_label = self._alma_panel_labeling
        else:
            raise Exception("Unknown panel labeling: "+self.telescope.panel_numbering)
        self._build_polar()
        self._build_ring_panels()
        self._build_ring_mask()
        self.fetch_panel = self._fetch_panel_ringed
        self.compile_panel_points = self._compile_panel_points_ringed
        self._nan_out_of_bounds = self._nan_out_of_bounds_ringed

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
        return '{0:d}-{1:d}'.format(iring+1, ipanel+1)

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
        angle = twopi/self.telescope.npanel[iring]
        sector_angle = twopi/self.telescope.npanel[0]
        theta = twopi-(ipanel+0.5)*angle
        sector = int(((theta/sector_angle)+1+self.telescope.npanel[0]/4) % self.telescope.npanel[0])
        if sector == 0:
            sector = self.telescope.npanel[0]
        nppersec = self.telescope.npanel[iring]/self.telescope.npanel[0]
        jpanel = int(nppersec-(ipanel % nppersec))
        return '{0:1d}-{1:1d}{2:1d}'.format(sector, iring+1, jpanel)

    def _crop_maps(self, margin=0.025):
        """
        Crop the amplitude and phase/deviation maps to decrease that usage and speedup calculations
        Args:
            margin: How much margin should be left outside the dish diameter
        """
        edge = (0.5+margin)*self.telescope.diam
        iumin = np.argmax(self.u_axis > -edge)
        iumax = np.argmax(self.u_axis > edge)
        ivmin = np.argmax(self.v_axis > -edge)
        ivmax = np.argmax(self.v_axis > edge)
        self.unpix = iumax-iumin
        self.vnpix = ivmax-ivmin
        self.u_axis = self.u_axis[iumin:iumax]
        self.v_axis = self.v_axis[ivmin:ivmax]
        self.amplitude = self.amplitude[iumin:iumax, ivmin:ivmax]
        if self.phase is not None:
            self.phase = self.phase[iumin:iumax, ivmin:ivmax]
        if self.deviation is not None:
            self.deviation = self.deviation[iumin:iumax, ivmin:ivmax]

    def _phase_to_deviation(self, phase):
        """
        Transforms a phase map to a physical deviation map
        Args:
            phase: Input phase map

        Returns:
            Physical deviation map
        """
        acoeff = (self.wavelength / twopi) / (4.0 * self.telescope.focus)
        bcoeff = 4 * self.telescope.focus ** 2
        return acoeff * phase * np.sqrt(self.rad ** 2 + bcoeff)

    def _deviation_to_phase(self, deviation):
        """
        Transforms a physical deviation map to a phase map
        Args:
            deviation: Input physical deviation map

        Returns:
            Phase map
        """
        acoeff = (self.wavelength / twopi) / (4.0 * self.telescope.focus)
        bcoeff = 4 * self.telescope.focus ** 2
        return deviation / (acoeff * np.sqrt(self.rad ** 2 + bcoeff))

    def _build_ring_mask(self):
        """
        Builds the mask on regions to be included in panel surface masks, specific to circular antennas as there is an
        outer and inner limit to the mask based on the antenna's inner receiver hole and outer edge
        """
        self.mask = np.where(self.amplitude < self.cut, False, True)
        self.mask = np.where(self.rad > self.telescope.inlim, self.mask, False)
        self.mask = np.where(self.rad < self.telescope.oulim, self.mask, False)
        self.mask = np.where(np.isnan(self.amplitude), False, self.mask)
        self.mask = np.where(self.deviation != self.deviation, False, self.mask)

    def _nan_out_of_bounds_ringed(self, data):
        """
        Replace by NaNs all data that is beyond the edges of the antenna surface
        Args:
            data: The array to be transformed

        Returns:
            Transformed array with nans outside the antenna valid limits
        """
        ouradius = np.where(self.rad > self.telescope.diam/2., np.nan, data)
        inradius = np.where(self.rad < self.telescope.inlim, np.nan, ouradius)
        return inradius

    def _build_polar(self):
        """
        Build polar coordinate grid, specific for circular antennas with panels arranged in rings
        """
        u2d = self.u_axis.reshape(self.unpix, 1)
        v2d = self.v_axis.reshape(1, self.vnpix)
        self.rad = np.sqrt(u2d**2 + v2d**2)
        self.phi = np.arctan2(u2d, -v2d)-pi/2
        self.phi = np.where(self.phi < 0, self.phi+twopi, self.phi)

    def _build_ring_panels(self):
        """
        Build list of panels, specific for circular antennas with panels arranged in rings
        """
        self.panels = []
        for iring in range(self.telescope.nrings):
            angle = 2.0 * np.pi / self.telescope.npanel[iring]
            for ipanel in range(self.telescope.npanel[iring]):

                panel = RingPanel(
                    self.panelmodel,
                    angle,
                    ipanel,
                    self._panel_label(iring, ipanel),
                    self.telescope.inrad[iring],
                    self.telescope.ourad[iring],
                    margin=self.panel_margins,
                    screw_scheme=self.telescope.screw_description,
                    screw_offset=self.telescope.screw_offset,
                    plot_screw_size=0.006 * self.telescope.diam
                )
                self.panels.append(panel)
        return

    def _compile_panel_points_ringed(self):
        panels = np.zeros(self.rad.shape)
        panelsum = 0
        for iring in range(self.telescope.nrings):
            angle = twopi/self.telescope.npanel[iring]
            panels = np.where(self.rad >= self.telescope.inrad[iring], np.floor(self.phi/angle) + panelsum, panels)
            panelsum += self.telescope.npanel[iring]
        panels = np.where(self.mask, panels, -1).astype("int32")
        for ix in range(self.unpix):
            xc = self.u_axis[ix]
            for iy in range(self.vnpix):
                ipanel = panels[ix, iy]
                if ipanel >= 0:
                    yc = self.v_axis[iy]
                    panel = self.panels[ipanel]
                    issample, inpanel = panel.is_inside(self.rad[ix, iy], self.phi[ix, iy])
                    if inpanel:
                        if issample:
                            panel.add_sample([xc, yc, ix, iy, self.deviation[ix, iy]])
                        else:
                            panel.add_margin([xc, yc, ix, iy, self.deviation[ix, iy]])
        self.panel_distribution = panels

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
            ipanel = np.sum(self.telescope.npanel[: ring - 1]) + panel - 1
        return self.panels[ipanel]

    def gains(self):
        """
        Computes antenna gains in decibels before and after panel surface fitting
        Returns:
        Gains before panel fitting OR Gains before and after panel fitting
        """
        self.ingains = self._gains_array(self.phase)
        if self.residuals is None:
            return self.ingains
        else:
            self.ougains = self._gains_array(self.phase_residuals)
            return self.ingains, self.ougains

    def _gains_array(self, arr):
        """
        Worker for gains method, works with the actual arrays to compute the gains
        This numpy version is significantly faster than the previous version
        Args:
            arr: Deviation image over which to compute the gains

        Returns:
        Actual and theoretical gains
        """
        thgain = fourpi * (1000.0 * self.reso / self.wavelength) ** 2
        gain = thgain * np.sqrt(np.sum(np.cos(arr[self.mask]))**2 + np.sum(np.sin(arr[self.mask]))**2)/np.sum(self.mask)
        return _convert_to_db(gain), _convert_to_db(thgain)

    def get_rms(self, unit='mm'):
        """
        Computes antenna surface RMS before and after panel surface fitting
        Returns:
        RMS before panel fitting OR RMS before and after panel fitting
        """
        fac = _convert_unit('m', unit, 'length')
        self.in_rms = self._compute_rms_array(self.deviation)
        if self.residuals is None:
            return fac*self.in_rms
        else:
            self.out_rms = self._compute_rms_array(self.residuals)
        return fac*self.in_rms, fac*self.out_rms

    def _compute_rms_array(self, array):
        """
        Factorized the computation of the RMS of an array
        Args:
            array: Input data array

        Returns:
            RMS of the input array
        """
        return np.sqrt(np.mean(array[self.mask] ** 2))

    def fit_surface(self):
        """
        Loops over the panels to fit the panel surfaces
        """
        panels = []
        for panel in self.panels:
            if not panel.solve():
                panels.append(panel.label)
        self.fitted = True
        if len(panels) > 0:
            msg = f'Fit failed with the {self.panelmodel} model and a simple mean has been used instead for the ' \
                  f'following panels: ' + str([self.antenna_name, self.ddi])
            logger = _get_astrohack_logger()
            logger.warning(msg)
            msg = str(panels)
            logger.warning(msg)

    def correct_surface(self):
        """
        Apply corrections determined by the panel surface fitting methods to the antenna surface
        """
        if not self.fitted:
            raise Exception("Panels must be fitted before atempting a correction")
        self.corrections = np.where(self.mask, 0, np.nan)
        self.residuals = np.copy(self.deviation)
        for panel in self.panels:
            corrections = panel.get_corrections()
            for corr in corrections:
                ix, iy = int(corr[0]), int(corr[1])
                self.residuals[ix, iy] -= corr[-1]
                self.corrections[ix, iy] = -corr[-1]
        self.phase_corrections = self._deviation_to_phase(self.corrections)
        self.phase_residuals = self._deviation_to_phase(self.residuals)
        self._build_panel_data_arrays()
        self.solved = True

    def print_misc(self):
        """
        Print miscelaneous information on the panels in the antenna surface
        """
        for panel in self.panels:
            panel.print_misc()

    def plot_mask(self, basename, screws=False, colormap=None, figuresize=None, dpi=300):
        """
        Plot mask used in the selection of points to be fitted
        Args:
            basename: basename for the plot, the prefix 'ancillary_mask' will be added to it
            screws: Are screw positions to be shown in plot?
            colormap: Colormap for amplitude plot
            figuresize: 2 element array with the image sizes in inches
            dpi: Plot resolution
        """
        plotmask = np.where(self.mask, 1, np.nan)
        plotname = _add_prefix(basename, 'mask')
        self._plot_map(plotname, plotmask, 'Mask', 0, 1, None, screws=screws, colormap=colormap, figuresize=figuresize,
                       dpi=dpi, colorbar=False)

    def plot_amplitude(self, basename, screws=False, colormap=None, figuresize=None, dpi=300):
        """
        Plot Amplitude map
        Args:
            basename: basename for the plot, the prefix 'ancillary_amplitude' will be added to it
            screws: Are screw positions to be shown in plot?
            colormap: Colormap for amplitude plot
            figuresize: 2 element array with the image sizes in inches
            dpi: Plot resolution
        """
        vmin, vmax = np.nanmin(self.amplitude), np.nanmax(self.amplitude)
        title = "Amplitude min={0:.5f}, max ={1:.5f} V".format(vmin, vmax)
        plotname = _add_prefix(basename, 'amplitude')
        self._plot_map(plotname, self.amplitude, title, vmin, vmax, self.amp_unit, screws=screws, colormap=colormap,
                       figuresize=figuresize, dpi=dpi)

    def plot_phase(self, basename, screws=False, colormap=None, figuresize=None, dpi=300, unit=None):
        """
        Plot phase map(s)
        Args:
            basename: basename for the plot(s), the prefix 'phase_{original|corrections|residuals}' will be added to it/them
            screws: Are screw positions to be shown in plot(s)?
            colormap: Colormap for phase plots
            figuresize: 2 element array with the image sizes in inches
            dpi: Plot resolution
            unit: Angle unit for plot(s)
        """
        if unit is None:
            unit = 'deg'
        fac = _convert_unit('rad', unit, 'trigonometric')
        prefix = 'phase'
        if self.residuals is None:
            maps = [self.phase]
            labels = ['original']
        else:
            maps = [self.phase, self.phase_corrections, self.phase_residuals]
            labels = ['original', 'corrections', 'residuals']
        self._multi_plot(maps, labels, prefix, basename, unit, fac, screws, colormap, figuresize, dpi)

    def plot_deviation(self, basename, screws=False, colormap=None, figuresize=None, dpi=300, unit=None):
        """
        Plot deviation map(s)
        Args:
            basename: basename for the plot(s), the prefix 'deviation_{original|corrections|residuals}' will be added to it/them
            screws: Are screw positions to be shown in plot(s)?
            colormap: Colormap for deviation plots
            figuresize: 2 element array with the image sizes in inches
            dpi: Plot resolution
            unit: Length unit for plot(s)
        """
        if unit is None:
            unit = 'mm'
        fac = _convert_unit('m', unit, 'length')
        prefix = 'deviation'
        if self.residuals is None:
            maps = [self.deviation]
            labels = ['original']
        else:
            maps = [self.deviation, self.corrections, self.residuals]
            labels = ['original', 'corrections', 'residuals']
        self._multi_plot(maps, labels, prefix, basename, unit, fac, screws, colormap, figuresize, dpi)

    def _multi_plot(self, maps, labels, prefix, basename, unit, conversion, screws, colormap=None, figuresize=None,
                    dpi=300):
        if len(maps) != len(labels):
            raise Exception('Map list and label list must be of the same size')
        nplots = len(maps)
        vmax = np.nanmax(np.abs(conversion*maps[0]))
        vmin = -vmax
        for iplot in range(nplots):
            title = f'{prefix.capitalize()} {labels[iplot]}'
            plotname = _add_prefix(basename, labels[iplot])
            plotname = _add_prefix(plotname, prefix)
            self._plot_map(plotname, conversion*maps[iplot], title, vmin, vmax, unit, screws=screws, dpi=dpi,
                           colormap=colormap, figuresize=figuresize)

    def _plot_map(self, filename, data, title, vmin, vmax, unit, screws=False, colormap=None, figuresize=None, dpi=300,
                  colorbar=True):
        if colormap is None:
            colormap = 'viridis'
        if figuresize is None:
            figuresize = figsize
        if figuresize is None or figuresize == 'None':
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figuresize)
        ax.set_title(title)
        # set the limits of the plot to the limits of the data
        xmin = np.min(self.u_axis)
        xmax = np.max(self.u_axis)
        ymin = np.min(self.v_axis)
        ymax = np.max(self.v_axis)
        im = ax.imshow(data, cmap=colormap, interpolation="nearest", extent=[xmin, xmax, ymin, ymax],
                       vmin=vmin, vmax=vmax,)
        if colorbar:
            _well_positioned_colorbar(ax, fig, im, "Z Scale [" + unit + "]")
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")
        for panel in self.panels:
            panel.plot(ax, screws=screws)
        fig.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()

    def plot_screw_adjustments(self, filename, unit, threshold=None, colormap=None, figuresize=None, dpi=300):
        """
        Plot screw adjustments as circles over a blank canvas with the panel layout
        Args:
            filename: Name of the output filename for the plot
            unit: Unit to display the screw adjustments
            threshold: Threshold below which data is considered negligable, value is assumed to be in the same unit as the plot, if not given defaults to 10% of the maximal deviation
            colormap: Colormap to display the screw adjustments
            figuresize: 2 element array with the image sizes in inches
            dpi: Resolution in pixels per inch
        """
        if colormap is None:
            cmap = cmaps['RdBu_r']
        else:
            cmap = cmaps[colormap]
        if figuresize is None or figuresize == 'None':
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figuresize)
        fac = _convert_unit('m', unit, 'length')
        vmax = np.nanmax(np.abs(fac * self.screw_adjustments))
        vmin = -vmax
        if threshold is None:
            threshold = 0.1*vmax
        else:
            threshold = np.abs(threshold)

        fig.suptitle('Screw corrections', y=0.92, fontsize='large')
        ax.set_title(f'\nThreshold = {threshold:.2f} {unit}', fontsize='small')
        # set the limits of the plot to the limits of the data
        xmin = np.min(self.u_axis)
        xmax = np.max(self.u_axis)
        ymin = np.min(self.v_axis)
        ymax = np.max(self.v_axis)
        im = ax.imshow(np.full_like(self.deviation, fill_value=np.nan), cmap=cmap, interpolation="nearest",
                       extent=[xmin, xmax, ymin, ymax], vmin=vmin, vmax=vmax)
        colorbar = _well_positioned_colorbar(ax, fig, im, "Screw adjustments [" + unit + "]")
        if threshold > 0:
            line = threshold
            while line < vmax:
                colorbar.ax.axhline(y=line, color='black', linestyle='-', lw=0.2)
                colorbar.ax.axhline(y=-line, color='black', linestyle='-', lw=0.2)
                line += threshold
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")

        for ipanel in range(len(self.panels)):
            self.panels[ipanel].plot(ax, screws=False)
            self.panels[ipanel].plot_corrections(ax, cmap, fac*self.screw_adjustments[ipanel], threshold, vmin, vmax)
        fig.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()

    def _build_panel_data_arrays(self):
        """
        Build arrays with data from the panels so that they can be stored on the XDS
        Returns:
            List with panel labels, panel fitting parameters, screw_adjustments
        """
        npanels = len(self.panels)
        NPAR = self.panels[0].NPAR
        nscrews = self.panels[0].screws.shape[0]
        self.panel_labels = np.ndarray([npanels], dtype=object)
        self.panel_pars = np.ndarray((npanels, NPAR), dtype=float)
        self.screw_adjustments = np.ndarray((npanels, nscrews), dtype=float)
        for ipanel in range(npanels):
            self.panel_labels[ipanel] = self.panels[ipanel].label
            self.panel_pars[ipanel, :] = self.panels[ipanel].par
            self.screw_adjustments[ipanel, :] = self.panels[ipanel].export_screws(unit='m')

    def export_screws(self, filename, unit="mm"):
        """
        Export screw adjustments for all panels onto an ASCII file
        Args:
            filename: ASCII file name/path
            unit: unit for panel screw adjustments ['mm','miliinches']
        """
        outfile =  "# Screw adjustments for {0:s} {1:s} antenna\n".format(self.telescope.name, self.antenna_name)
        outfile += "# Adjustments are in " + unit + 2*lnbr
        outfile += "# Lower means away from subreflector" + lnbr
        outfile += "# Raise means toward the subreflector" + lnbr
        outfile += "# LOWER the panel if the number is POSITIVE" + lnbr
        outfile += "# RAISE the panel if the number is NEGATIVE" + lnbr
        outfile += 2 * lnbr
        outfile += "{0:16s}".format('Panel')
        nscrews = len(self.telescope.screw_description)
        for screw in self.telescope.screw_description:
            outfile += "{0:11s}".format(screw)
        outfile += lnbr
        fac = _convert_unit('m', unit, 'length')
        for ipanel in range(len(self.panel_labels)):
            outfile += "{0:8s}".format(self.panel_labels[ipanel])
            for iscrew in range(nscrews):
                outfile += " {0:10.2f}".format(fac*self.screw_adjustments[ipanel, iscrew])
            outfile += lnbr

        lefile = open(filename, "w")
        lefile.write(outfile)
        lefile.close()

    def export_xds(self):
        """
        Export all the data to Xarray dataset
        Returns:
            XarrayDataSet contaning all the relevant information
        """
        xds = xr.Dataset()
        gains = self.gains()
        rms = self.get_rms(unit='m')
        xds.attrs['telescope_name'] = self.telescope.name
        xds.attrs['ant_name'] = self.antenna_name
        xds.attrs['ddi'] = self.ddi
        xds.attrs['wavelength'] = self.wavelength
        xds.attrs['amp_unit'] = self.amp_unit
        xds.attrs['panel_model'] = self.panelmodel
        xds.attrs['panel_margin'] = self.panel_margins
        xds.attrs['cutoff'] = self.cut
        xds.attrs['solved'] = self.solved
        xds.attrs['fitted'] = self.fitted
        xds['AMPLITUDE'] = xr.DataArray(self.amplitude, dims=["u", "v"])
        xds['PHASE'] = xr.DataArray(self.phase, dims=["u", "v"])
        xds['DEVIATION'] = xr.DataArray(self.deviation, dims=["u", "v"])
        xds['MASK'] = xr.DataArray(self.mask, dims=["u", "v"])
        xds['PANEL_DISTRIBUTION'] = xr.DataArray(self.panel_distribution, dims=["u", "v"])
        if self.solved:
            xds['PHASE_RESIDUALS'] = xr.DataArray(self.phase_residuals, dims=["u", "v"])
            xds['RESIDUALS'] = xr.DataArray(self.residuals, dims=["u", "v"])
            xds['PHASE_CORRECTIONS'] = xr.DataArray(self.phase_corrections, dims=["u", "v"])
            xds['CORRECTIONS'] = xr.DataArray(self.corrections, dims=["u", "v"])
            xds.attrs['input_rms'] = rms[0]
            xds.attrs['output_rms'] = rms[1]
            xds.attrs['input_gain'] = gains[0][0]
            xds.attrs['output_gain'] = gains[1][0]
            xds.attrs['theoretical_gain'] = gains[0][1]
            xds['PANEL_PARAMETERS'] = xr.DataArray(self.panel_pars, dims=['labels', 'pars'])
            xds['PANEL_SCREWS'] = xr.DataArray(self.screw_adjustments, dims=['labels', 'screws'])
            coords = {"u": self.u_axis, "v": self.v_axis, 'labels': self.panel_labels,
                      'screws': self.telescope.screw_description, 'pars': np.arange(self.panel_pars.shape[1])}
        else:
            xds.attrs['input_rms'] = rms
            xds.attrs['input_gain'] = gains[0]
            xds.attrs['theoretical_gain'] = gains[1]
            coords = {"u": self.u_axis, "v": self.v_axis}

        xds = xds.assign_coords(coords)
        return xds

    def export_to_fits(self, basename):
        """
        Data to export: Amplitude, mask, phase, phase_corrections, phase_residuals, deviations, deviation_corrections, deviation_residuals
        conveniently all data are on the same grid!
        Returns:
        """

        head = {
            'PMODEL'  : self.panelmodel,
            'PMARGIN' : self.panel_margins,
            'CUTOFF'  : self.cut,
            'TELESCOP': self.antenna_name,
            'INSTRUME': self.telescope.name,
            'WAVELENG': self.wavelength,
            'FREQUENC': clight/self.wavelength
        }
        head = _axis_to_fits_header(head, self.u_axis, 1, 'X', 'm')
        head = _axis_to_fits_header(head, self.v_axis, 2, 'Y', 'm')

        _write_fits(head, 'Amplitude', self.amplitude, basename + '_amplitude.fits', self.amp_unit, 'panel')
        _write_fits(head, 'Mask', np.where(self.mask, 1.0, np.nan), basename + '_mask.fits', '', 'panel')
        _write_fits(head, 'Original Phase', self.phase, basename + '_phase_original.fits', 'rad', 'panel')
        _write_fits(head, 'Phase Corrections', self.phase_corrections, basename + '_phase_correction.fits', 'rad',
                    'panel')
        _write_fits(head, 'Phase residuals', self.phase_residuals, basename + '_phase_residual.fits', 'rad',
                    'panel')
        _write_fits(head, 'Original Deviation', self.deviation, basename + '_deviation_original.fits', 'm',
                    'panel')
        _write_fits(head, 'Deviation Corrections', self.corrections, basename + '_deviation_correction.fits', 'm',
                    'panel')
        _write_fits(head, 'Deviation residuals', self.residuals, basename + '_deviation_residual.fits', 'm',
                    'panel')

