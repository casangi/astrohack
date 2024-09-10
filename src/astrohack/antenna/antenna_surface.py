import xarray as xr

from matplotlib import patches

import toolviper.utils.logger as logger

from astrohack.antenna.ring_panel import RingPanel
from astrohack.utils import string_to_ascii_file, create_dataset_label
from astrohack.utils.constants import *
from astrohack.utils.conversion import to_db
from astrohack.utils.conversion import convert_unit
from astrohack.utils.text import add_prefix, bool_to_str, format_frequency
from astrohack.visualization.plot_tools import well_positioned_colorbar, create_figure_and_axes, close_figure, \
    get_proper_color_map

from astrohack.utils.fits import write_fits, resolution_to_fits_header, axis_to_fits_header

lnbr = "\n"
SUPPORTED_POL_STATES = ['I', 'RR', 'LL', 'XX', 'YY']


class AntennaSurface:
    def __init__(self, inputxds, telescope, clip_type='sigma', clip_level=3, pmodel='rigid', crop=False,
                 nan_out_of_bounds=True, panel_margins=0.05, reread=False, pol_state='I'):
        """
        Antenna Surface description capable of computing RMS, Gains, and fitting the surface to obtain screw adjustments
        Args:
            inputxds: Input xarray dataset
            telescope: Telescope object
            clip_type: Type of clipping to be applied to amplitude
            clip_level: Level of clipping
            pmodel: model of panel surface fitting, if is None defaults to telescope default
            crop: Crop apertures to slightly larger frames than the antenna diameter
            nan_out_of_bounds: Should the region outside the dish be replaced with NaNs?
            panel_margins: Margin to be ignored at edges of panels when fitting, defaults to 20% if None
            reread: Read a previously processed holography
        """
        self.reread = reread
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
        self.pol_state = pol_state
        self._read_xds(inputxds)
        self.telescope = telescope

        if not self.reread:
            self.panelmodel = pmodel
            self.panel_margins = panel_margins
            if crop:
                self._crop_maps()

        if self.telescope.ringed:
            self._init_ringed(clip_type, clip_level)
        if not self.reread:
            if self.computephase:
                self.phase = self._deviation_to_phase(self.deviation)
            else:
                self.deviation = self._phase_to_deviation(self.phase)

            if nan_out_of_bounds:
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
        self.resolution = None

    def _read_holog_xds(self, inputxds):
        if 'chan' in inputxds.dims:
            if inputxds.sizes['chan'] != 1:
                raise Exception("Only single channel holographies supported")
            self.wavelength = clight / inputxds.chan.values[0]
        else:
            self.wavelength = inputxds.attrs['wavelength']

        if self.pol_state not in inputxds.coords['pol']:
            msg = f'Polarization state {self.pol_state} is not present in the data (available states: ' \
                  f'{inputxds.coords["pol"]})'
            logger.error(msg)
            raise Exception(msg)

        self.amplitude = inputxds["AMPLITUDE"].sel(pol=self.pol_state).isel(time=0, chan=0).values
        self.phase = inputxds["CORRECTED_PHASE"].sel(pol=self.pol_state).isel(time=0, chan=0).values

        self.npoint = np.sqrt(inputxds.sizes['l'] ** 2 + inputxds.sizes['m'] ** 2)
        self.amp_unit = 'V'
        self.u_axis = inputxds.u_prime.values
        self.v_axis = inputxds.v_prime.values
        self.computephase = False

        try:
            self.resolution = inputxds.attrs['aperture_resolution']
        except KeyError:

            logger.warning("holog image does not have resolution information")
            logger.warning("Rerun holog with astrohack v>0.1.5 for aperture resolution information")
            self.resolution = None

    def _read_panel_xds(self, inputxds):
        self.wavelength = inputxds.attrs['wavelength']
        self.amp_unit = inputxds.attrs['amp_unit']
        self.panelmodel = inputxds.attrs['panel_model']
        self.panel_margins = inputxds.attrs['panel_margin']
        self.clip = inputxds.attrs['clip']
        self.solved = inputxds.attrs['solved']
        self.fitted = inputxds.attrs['fitted']
        self.pol_state = inputxds.attrs['pol_state']
        # Arrays
        self.amplitude = inputxds['AMPLITUDE'].values
        self.phase = inputxds['PHASE'].values
        self.deviation = inputxds['DEVIATION'].values
        self.mask = inputxds['MASK']
        self.u_axis = inputxds.u.values
        self.v_axis = inputxds.u.values
        self.panel_distribution = inputxds['PANEL_DISTRIBUTION'].values
        try:
            self.resolution = inputxds.attrs['aperture_resolution']
        except KeyError:

            logger.warning("Input panel file does not have resolution information")
            logger.warning("Rerun holog with astrohack v>0.1.5 for aperture resolution information")
            self.resolution = None

        if self.solved:
            self.panel_fallback = inputxds['PANEL_FALLBACK'].values
            self.panel_model_array = inputxds['PANEL_MODEL'].values
            self.phase_residuals = inputxds['PHASE_RESIDUALS'].values
            self.residuals = inputxds['RESIDUALS'].values
            self.phase_corrections = inputxds['PHASE_CORRECTIONS'].values
            self.corrections = inputxds['CORRECTIONS'].values
            self.panel_pars = inputxds['PANEL_PARAMETERS'].values
            self.screw_adjustments = inputxds['PANEL_SCREWS'].values
            self.ingains = [inputxds.attrs['input_gain'], inputxds.attrs['theoretical_gain']]
            self.ougains = [inputxds.attrs['output_gain'], inputxds.attrs['theoretical_gain']]
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
        self.label = create_dataset_label(inputxds.attrs['ant_name'], inputxds.attrs['ddi'])

    def _measure_ring_clip(self, clip_type, clip_level):
        if clip_type == 'relative':
            clip = clip_level * np.nanmax(self.amplitude)
        elif clip_type == 'absolute':
            clip = clip_level
        elif clip_type == 'sigma':
            noise = np.where(self.rad < self.telescope.diam / 2., np.nan, self.amplitude)
            noiserms = np.sqrt(np.nanmean(noise ** 2))
            clip = clip_level * noiserms
        else:
            msg = f'Unrecognized clipping type: {clip_type}'
            raise Exception(msg)
        return clip

    def _init_ringed(self, clip_type, clip_level):
        """
        Do the proper initialization and method association for the case of a ringed antenna
        """
        if self.telescope.panel_numbering == 'ring, clockwise, top':
            self._panel_label = self._vla_panel_labeling
        elif self.telescope.panel_numbering == 'sector, counterclockwise, right':
            self._panel_label = self._alma_panel_labeling
        else:
            raise Exception("Unknown panel labeling: " + self.telescope.panel_numbering)
        self._build_polar()
        if not self.reread:
            self.clip = self._measure_ring_clip(clip_type, clip_level)
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
        return '{0:d}-{1:d}'.format(iring + 1, ipanel + 1)

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
        angle = twopi / self.telescope.npanel[iring]
        sector_angle = twopi / self.telescope.npanel[0]
        theta = twopi - (ipanel + 0.5) * angle
        sector = int(((theta / sector_angle) + 1 + self.telescope.npanel[0] / 4) % self.telescope.npanel[0])
        if sector == 0:
            sector = self.telescope.npanel[0]
        nppersec = self.telescope.npanel[iring] / self.telescope.npanel[0]
        jpanel = int(nppersec - (ipanel % nppersec))
        return '{0:1d}-{1:1d}{2:1d}'.format(sector, iring + 1, jpanel)

    def _crop_maps(self, margin=0.025):
        """
        Crop the amplitude and phase/deviation maps to decrease that usage and speedup calculations
        Args:
            margin: How much margin should be left outside the dish diameter
        """
        edge = (0.5 + margin) * self.telescope.diam
        iumin = np.argmax(self.u_axis > -edge)
        iumax = np.argmax(self.u_axis > edge)
        ivmin = np.argmax(self.v_axis > -edge)
        ivmax = np.argmax(self.v_axis > edge)
        self.unpix = iumax - iumin
        self.vnpix = ivmax - ivmin
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
        self.mask = np.where(self.amplitude < self.clip, False, True)
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
        ouradius = np.where(self.rad > self.telescope.diam / 2., np.nan, data)
        inradius = np.where(self.rad < self.telescope.inlim, np.nan, ouradius)
        return inradius

    def _build_polar(self):
        """
        Build polar coordinate grid, specific for circular antennas with panels arranged in rings
        """
        u2d = self.u_axis.reshape(self.unpix, 1)
        v2d = self.v_axis.reshape(1, self.vnpix)
        self.rad = np.sqrt(u2d ** 2 + v2d ** 2)
        self.phi = np.arctan2(u2d, -v2d) - pi / 2
        self.phi = np.where(self.phi < 0, self.phi + twopi, self.phi)

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
        panels = np.full_like(self.rad, -1)
        panelsum = 0
        for iring in range(self.telescope.nrings):
            angle = twopi / self.telescope.npanel[iring]
            panels = np.where(self.rad >= self.telescope.inrad[iring], np.floor(self.phi / angle) + panelsum,
                              panels)
            panelsum += self.telescope.npanel[iring]
        for ix in range(self.unpix):
            xc = self.u_axis[ix]
            for iy in range(self.vnpix):
                ipanel = panels[ix, iy]
                if ipanel >= 0:
                    yc = self.v_axis[iy]
                    panel = self.panels[int(ipanel)]
                    issample, inpanel = panel.is_inside(self.rad[ix, iy], self.phi[ix, iy])
                    if inpanel:
                        if issample and self.mask[ix, iy]:
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
        self.ingains = self.gain_at_wavelength(False, self.wavelength)
        if not self.solved:
            return self.ingains

        else:
            self.ougains = self.gain_at_wavelength(True, self.wavelength)
            return self.ingains, self.ougains

    def gain_at_wavelength(self, corrected, wavelength):
        # This is valid for the VLA not sure if valid for anything else...
        wavelength_scaling = self.wavelength / wavelength

        dish_mask = np.where(self.rad > self.telescope.inlim, True, False)
        dish_mask = np.where(self.rad < self.telescope.diam/2, dish_mask, False)

        if corrected:
            if self.fitted:
                scaled_phase = wavelength_scaling*self.phase_residuals
            else:
                msg = 'Cannot computed gains for corrected dish if panels are not fitted.'
                logger.error(msg)
                raise Exception(msg)
        else:
            scaled_phase = wavelength_scaling*self.phase

        cossum = np.nansum(np.cos(scaled_phase[dish_mask]))
        sinsum = np.nansum(np.sin(scaled_phase[dish_mask]))
        real_factor = np.sqrt(cossum**2 + sinsum**2)/np.sum(dish_mask)

        u_fact = (self.u_axis[1] - self.u_axis[0]) / wavelength
        v_fact = (self.v_axis[1] - self.v_axis[0]) / wavelength

        theo_gain = fourpi * np.abs(u_fact * v_fact)
        real_gain = theo_gain * real_factor
        return to_db(real_gain), to_db(theo_gain)

    def get_rms(self, unit='mm'):
        """
        Computes antenna surface RMS before and after panel surface fitting
        Returns:
        RMS before panel fitting OR RMS before and after panel fitting
        """
        fac = convert_unit('m', unit, 'length')
        self.in_rms = self._compute_rms_array(self.deviation)
        if self.residuals is None:
            return fac * self.in_rms
        else:
            self.out_rms = self._compute_rms_array(self.residuals)
        return fac * self.in_rms, fac * self.out_rms

    def _compute_rms_array(self, array):
        """
        Factorized the computation of the RMS of an array
        Args:
            array: Input data array

        Returns:
            RMS of the input array
        """
        return np.sqrt(np.nanmean(array[self.mask] ** 2))

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
            msg = (f'{self.label}: Fit failed with the {self.panelmodel} model and a simple mean has been used instead '
                   f'for the following panels:')

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

    def plot_mask(self, basename, caller, parm_dict):
        """
        Plot mask used in the selection of points to be fitted
        Args:
            basename: basename for the plot, the prefix 'ancillary_mask' will be added to it
            caller: Which mds called this plotting function
            parm_dict: dictionary with plotting parameters
        """
        plotmask = np.where(self.mask, 1, np.nan)
        plotname = add_prefix(basename, f'{caller}_mask')
        parm_dict['z_lim'] = [0, 1]
        parm_dict['unit'] = ' '
        self._plot_map(plotname, plotmask, 'Mask', parm_dict, colorbar=False)

    def plot_amplitude(self, basename, caller, parm_dict):
        """
        Plot Amplitude map
        Args:
            basename: basename for the plot, the prefix 'ancillary_amplitude' will be added to it
            caller: Which mds called this plotting function
            parm_dict: dictionary with plotting parameters
        """
        if parm_dict['amplitude_limits'] is None or parm_dict['amplitude_limits'] == "None":
            parm_dict['z_lim'] = np.nanmin(self.amplitude), np.nanmax(self.amplitude)
        else:
            parm_dict['z_lim'] = parm_dict['amplitude_limits']

        title = "Amplitude, min={0:.5f}, max ={1:.5f} V".format(parm_dict['z_lim'][0], parm_dict['z_lim'][1])
        plotname = add_prefix(basename, f'{caller}_amplitude')
        parm_dict['unit'] = self.amp_unit
        self._plot_map(plotname, self.amplitude, title, parm_dict)

    def plot_phase(self, basename, caller, parm_dict):
        """
        Plot phase map(s)
        Args:
            basename: basename for the plot(s), the prefix 'phase_{original|corrections|residuals}' will be added to
                      it/them
            caller: Which mds called this plotting function
            parm_dict: dictionary with plotting parameters
        """

        if parm_dict['phase_unit'] is None:
            parm_dict['unit'] = 'deg'
        else:
            parm_dict['unit'] = parm_dict['phase_unit']
        parm_dict['z_lim'] = parm_dict['phase_limits']
        fac = convert_unit('rad', parm_dict['unit'], 'trigonometric')
        prefix = 'phase'
        if caller == 'image':
            prefix = 'corrected'
            maps = [self.phase]
            labels = ['phase']
        else:
            if self.residuals is None:
                maps = [self.phase]
                labels = ['original']
            else:
                maps = [self.phase, self.phase_corrections, self.phase_residuals]
                labels = ['original', 'correction', 'residual']
        self._multi_plot(maps, labels, prefix, basename, fac, parm_dict, caller)

    def plot_deviation(self, basename, caller, parm_dict):
        """
        Plot deviation map(s)
        Args:
            basename: basename for the plot(s), the prefix 'deviation_{original|corrections|residuals}' will be added
                      to it/them
            caller: Which mds called this plotting function
            parm_dict: dictionary with plotting parameters
        """
        if parm_dict['deviation_unit'] is None:
            parm_dict['unit'] = 'mm'
        else:
            parm_dict['unit'] = parm_dict['deviation_unit']
        parm_dict['z_lim'] = parm_dict['deviation_limits']
        fac = convert_unit('m', parm_dict['unit'], 'length')
        prefix = 'deviation'
        rms = self.get_rms(unit=parm_dict['unit'])
        if caller == 'image':
            prefix = 'original'
            maps = [self.deviation]
            labels = ['deviation']
        else:
            if self.residuals is None:
                maps = [self.deviation]
                labels = [f'original RMS={rms:.2f} {parm_dict["unit"]}']
            else:
                maps = [self.deviation, self.corrections, self.residuals]
                labels = [f'original RMS={rms[0]:.2f} {parm_dict["unit"]}', 'correction', f'residual RMS={rms[1]:.2f} '
                                                                                          f'{parm_dict["unit"]}']
        self._multi_plot(maps, labels, prefix, basename, fac, parm_dict, caller)

    def _multi_plot(self, maps, labels, prefix, basename, factor, parm_dict, caller):
        if len(maps) != len(labels):
            raise Exception('Map list and label list must be of the same size')
        nplots = len(maps)
        if parm_dict['z_lim'] is None or parm_dict['z_lim'] == "None":
            vmax = np.nanmax(np.abs(factor * maps[0]))  # Gotten from the original map (displays the biggest variation)
            parm_dict['z_lim'] = [-vmax, vmax]
        for iplot in range(nplots):
            title = f'{prefix.capitalize()} {labels[iplot]}'
            plotname = add_prefix(basename, labels[iplot].split()[0])
            plotname = add_prefix(plotname, prefix)
            plotname = add_prefix(plotname, caller)
            self._plot_map(plotname, factor * maps[iplot], title, parm_dict)

    def _plot_map(self, filename, data, title, parm_dict, colorbar=True):
        cmap = get_proper_color_map(parm_dict['colormap'])
        fig, ax = create_figure_and_axes(parm_dict['figure_size'], [1, 1])
        ax.set_title(title)
        # set the limits of the plot to the limits of the data
        extent = [np.min(self.u_axis), np.max(self.u_axis), np.min(self.v_axis), np.max(self.v_axis)]
        vmin, vmax = parm_dict['z_lim']
        im = ax.imshow(data, cmap=cmap, interpolation="nearest", extent=extent,
                       vmin=vmin, vmax=vmax, )
        self._add_resolution_to_plot(ax, extent)
        if colorbar:
            well_positioned_colorbar(ax, fig, im, "Z Scale [" + parm_dict['unit'] + "]")

        self._add_resolution_to_plot(ax, extent)
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")
        for panel in self.panels:
            panel.plot(ax, screws=parm_dict['plot_screws'], label=parm_dict['panel_labels'])

        suptitle = f'{self.label}, Pol. state: {self.pol_state}'
        close_figure(fig, suptitle, filename, parm_dict['dpi'], parm_dict['display'])

    def _add_resolution_to_plot(self, ax, extent, xpos=0.9, ypos=0.1):
        lw = 0.5
        if self.resolution is None:
            return
        dx = extent[1] - extent[0]
        dy = extent[3] - extent[2]
        center = (extent[0] + xpos * dx, extent[2] + ypos * dy)
        resolution = patches.Ellipse(center, self.resolution[0], self.resolution[1], angle=0.0, linewidth=lw,
                                     color='black', zorder=2, fill=False)
        ax.add_patch(resolution)
        halfbeam = self.resolution / dy / 2
        ax.axvline(x=center[0], ymin=ypos - halfbeam[1], ymax=ypos + halfbeam[1], color='black', lw=lw / 2)
        ax.axhline(y=center[1], xmin=xpos - halfbeam[0], xmax=xpos + halfbeam[0], color='black', lw=lw / 2)

    def plot_screw_adjustments(self, filename, parm_dict):
        """
        Plot screw adjustments as circles over a blank canvas with the panel layout
        Args:
            filename: Name of the output filename for the plot
            parm_dict: Dictionary with plotting parameters
        """
        unit = parm_dict['unit']
        threshold = parm_dict['threshold']
        cmap = get_proper_color_map(parm_dict['colormap'], default_cmap='RdBu_r')
        fig, ax = create_figure_and_axes(parm_dict['figure_size'], [1, 1])

        fac = convert_unit('m', unit, 'length')
        vmax = np.nanmax(np.abs(fac * self.screw_adjustments))
        vmin = -vmax
        if threshold is None or threshold == 'None':
            threshold = 0.1 * vmax
        else:
            threshold = np.abs(threshold)

        ax.set_title(f'\nThreshold = {threshold:.2f} {unit}', fontsize='small')
        # set the limits of the plot to the limits of the data
        extent = [np.min(self.u_axis), np.max(self.u_axis), np.min(self.v_axis), np.max(self.v_axis)]
        im = ax.imshow(np.full_like(self.deviation, fill_value=np.nan), cmap=cmap, interpolation="nearest",
                       extent=extent, vmin=vmin, vmax=vmax)

        self._add_resolution_to_plot(ax, extent)
        colorbar = well_positioned_colorbar(ax, fig, im, "Screw adjustments [" + unit + "]")
        if threshold > 0:
            line = threshold

            while line < vmax:
                colorbar.ax.axhline(y=line, color='black', linestyle='-', lw=0.2)
                colorbar.ax.axhline(y=-line, color='black', linestyle='-', lw=0.2)
                line += threshold

        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")

        for ipanel in range(len(self.panels)):
            self.panels[ipanel].plot(ax, screws=False, label=parm_dict['panel_labels'])
            self.panels[ipanel].plot_corrections(ax, cmap, fac * self.screw_adjustments[ipanel], threshold, vmin, vmax)

        suptitle = f'{self.label}, Pol. state: {self.pol_state}'
        close_figure(fig, suptitle, filename, parm_dict['dpi'], parm_dict['display'])

    def _build_panel_data_arrays(self):
        """
        Build arrays with data from the panels so that they can be stored on the XDS
        Returns:
            List with panel labels, panel fitting parameters, screw_adjustments
        """
        npanels = len(self.panels)
        # First panel might fail hence we need to check npar for all panels
        max_par = 0
        for panel in self.panels:
            p_npar = panel.model.npar
            if p_npar > max_par:
                max_par = p_npar

        nscrews = self.panels[0].screws.shape[0]

        self.panel_labels = np.ndarray([npanels], dtype=object)
        self.panel_model_array = np.ndarray([npanels], dtype=object)
        self.panel_pars = np.full((npanels, max_par), np.nan, dtype=float)
        self.screw_adjustments = np.ndarray((npanels, nscrews), dtype=float)
        self.panel_fallback = np.ndarray([npanels], dtype=bool)

        for ipanel in range(npanels):
            self.panel_labels[ipanel] = self.panels[ipanel].label
            self.panel_pars[ipanel, :] = self.panels[ipanel].model.parameters
            self.screw_adjustments[ipanel, :] = self.panels[ipanel].export_screws(unit='m')
            self.panel_model_array[ipanel] = self.panels[ipanel].model_name
            self.panel_fallback[ipanel] = self.panels[ipanel].fall_back_fit

    def export_screws(self, filename, unit="mm", comment_char='#'):
        """
        Export screw adjustments for all panels onto an ASCII file
        Args:
            filename: ASCII file name/path
            unit: unit for panel screw adjustments ['mm','miliinches']
            comment_char: Character used for comments
        """
        outfile = f"# Screw adjustments for {self.telescope.name}'s {self.label}, pol. state {self.pol_state}\n"
        freq = clight/self.wavelength
        out_freq = format_frequency(freq)
        outfile += f"# Frequency = {out_freq}{lnbr}"
        outfile += "# Adjustments are in " + unit + 2 * lnbr
        outfile += "# Lower means away from subreflector" + lnbr
        outfile += "# Raise means toward the subreflector" + lnbr
        outfile += "# LOWER the panel if the number is POSITIVE" + lnbr
        outfile += "# RAISE the panel if the number is NEGATIVE" + lnbr
        outfile += 2 * lnbr
        spc = ' '
        outfile += f'{comment_char} Panel{2*spc}'
        nscrews = len(self.telescope.screw_description)
        for screw in self.telescope.screw_description:
            outfile += f"{4*spc}{screw:2s}{4*spc}"
        outfile += f'Fallback{4*spc}Model{lnbr}'
        fac = convert_unit('m', unit, 'length')

        for ipanel in range(len(self.panel_labels)):
            outfile += "{0:>5s}".format(self.panel_labels[ipanel])

            for iscrew in range(nscrews):
                outfile += " {0:>9.2f}".format(fac * self.screw_adjustments[ipanel, iscrew])

            outfile += (f'{5*spc}{bool_to_str(self.panel_fallback[ipanel]):>3s}{7*spc}{self.panel_model_array[ipanel]}'
                        + lnbr)

        string_to_ascii_file(outfile, filename)

    def export_xds(self):
        """
        Export all the data to Xarray dataset
        Returns:
            XarrayDataSet containing all the relevant information
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
        xds.attrs['clip'] = self.clip
        xds.attrs['solved'] = self.solved
        xds.attrs['fitted'] = self.fitted
        xds.attrs['aperture_resolution'] = self.resolution
        xds.attrs['pol_state'] = self.pol_state
        xds['AMPLITUDE'] = xr.DataArray(self.amplitude, dims=["u", "v"])
        xds['PHASE'] = xr.DataArray(self.phase, dims=["u", "v"])
        xds['DEVIATION'] = xr.DataArray(self.deviation, dims=["u", "v"])
        xds['MASK'] = xr.DataArray(self.mask, dims=["u", "v"])
        xds['PANEL_DISTRIBUTION'] = xr.DataArray(self.panel_distribution, dims=["u", "v"])

        coords = {"u": self.u_axis,
                  "v": self.v_axis}

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
            xds['PANEL_MODEL'] = xr.DataArray(self.panel_model_array, dims=['labels'])
            xds['PANEL_FALLBACK'] = xr.DataArray(self.panel_fallback, dims=['labels'])

            coords = {**coords,
                      "labels": self.panel_labels,
                      "screws": self.telescope.screw_description,
                      "pars": np.arange(self.panel_pars.shape[1])
                      }

        else:
            xds.attrs['input_rms'] = rms
            xds.attrs['input_gain'] = gains[0]
            xds.attrs['theoretical_gain'] = gains[1]

        xds = xds.assign_coords(coords)
        return xds

    def export_to_fits(self, basename):
        """
        Data to export: Amplitude, mask, phase, phase_corrections, phase_residuals, deviations, deviation_corrections,
                        deviation_residuals
        conveniently all data are on the same grid!
        Returns:
        """

        head = {
            'PMODEL': self.panelmodel,
            'PMARGIN': self.panel_margins,
            'CLIP': self.clip,
            'TELESCOP': self.antenna_name,
            'INSTRUME': self.telescope.name,
            'WAVELENG': self.wavelength,
            'FREQUENC': clight / self.wavelength,
        }
        head = axis_to_fits_header(head, self.u_axis, 1, 'X----LIN', 'm')
        head = axis_to_fits_header(head, self.v_axis, 2, 'Y----LIN', 'm')
        head = resolution_to_fits_header(head, self.resolution)

        write_fits(head, 'Amplitude', self.amplitude, add_prefix(basename, 'amplitude') + '.fits', self.amp_unit,
                   'panel')
        write_fits(head, 'Mask', np.where(self.mask, 1.0, np.nan), add_prefix(basename, 'mask') + '.fits', '', 'panel')
        write_fits(head, 'Original Phase', self.phase, add_prefix(basename, 'phase_original') + '.fits', 'rad', 'panel')
        write_fits(head, 'Phase Corrections', self.phase_corrections,
                   add_prefix(basename, 'phase_correction') + '.fits', 'rad', 'panel')
        write_fits(head, 'Phase residuals', self.phase_residuals, add_prefix(basename, 'phase_residual') + '.fits',
                   'rad', 'panel')
        write_fits(head, 'Original Deviation', self.deviation, add_prefix(basename, 'deviation_original') + '.fits',
                   'm', 'panel')
        write_fits(head, 'Deviation Corrections', self.corrections,
                   add_prefix(basename, 'deviation_correction') + '.fits', 'm', 'panel')
        write_fits(head, 'Deviation residuals', self.residuals, add_prefix(basename, 'deviation_residual') + '.fits',
                   'm', 'panel')
