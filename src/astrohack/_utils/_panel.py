import numpy as np

from numba import njit

from astrohack._utils._conversion import _convert_unit
from astrohack._utils._algorithms import _least_squares_fit_block
from astrohack._utils._constants import plot_types

# global constants
NPAR = 10


def _phase_fitting_block(pols, wavelength, telescope, cellxy, amplitude_image, phase_image, pointing_offset,
                         focus_xy_offsets, focus_z_offset, subreflector_tilt, cassegrain_offset):
    """
    Corrects the grading phase for pointing, focus, and feed offset errors using the least squares method, and a model
    incorporating subreflector position errors.  Includes reference pointing

    This is a revised version of the task, offering a two-reflector solution.  M. Kesteven, 6/12/1994

    The formulation is in terms of the Ruze expressions (the unpublished
    lecture notes : Small Displacements in Parabolic Antennas, 1969).

    Code ported from AIPS subroutine fltph3 of the HOLOG task

    results and error arrays contain 10 values:
    0        Constant phase offset, in degrees.
    1        X direction phase ramp, in degress per cell
    2        Y direction phase ramp, in degrees per cell
    3        X direction focus offset, in mm
    4        Y direction focus offset, in mm
    5        Z direction focus offset, in mm
    6        X direction Subreflector tilt, in degrees
    7        Y direction subreflector tilt, in degrees
    8        X direction cassegrain offset in mm
    9        Y direction cassegrain offset in mm

    Based on AIPS code by:
        Mark Calabretta, Australia Telescope.
        Origin; 1987/Nov.    Code last modified; 1989/Nov/01.
        mjk, 28/1/93
        RAP, 27/05/08

    Args:
        pols: Indices of the polarizations to be used for phase fitting
        wavelength: Observing wavelength, in meters
        telescope: Telescope object containing the optics parameters
        cellxy: Map cell spacing, in meters
        amplitude_image: Grading amplitude map
        phase_image: Grading phase map
        pointing_offset: enable phase slope (pointing offset)
        focus_xy_offsets: enable subreflector offset model
        focus_z_offset: enable subreflector focus (z) model
        subreflector_tilt: Enable subreflector rotation model
        cassegrain_offset: enable Cassegrain offsets (X, Y, Z)

    Returns:
        results: Array containining the fit results in convenient units
        errors: Array containing the fit errors in convenient units
        corrected_phase: Phase map corrected for fitted parameters
        phase_model: Phase model used for the correction
        in_rms: Phase RMS before fitting
        out_rms: Phase RMS after fitting
    """
    matrix, vector = _build_design_matrix_block(pols, -telescope.inlim, -telescope.diam/2, cellxy, phase_image,
                                                amplitude_image, telescope.magnification, telescope.surp_slope,
                                                telescope.focus)

    ignored = _build_ignored_array(pointing_offset, focus_xy_offsets, focus_z_offset, subreflector_tilt,
                                   cassegrain_offset)
    matrix, vector = _ignore_non_fitted_block(ignored, matrix, vector)

    # #   compute the least squares solution.
    results, variances = _least_squares_fit_block(matrix, vector)
    #
    # Reconstruct full output for ignored parameters
    results, variances = _reconstruct_full_results_block(results, variances, ignored)
    #
    # apply the correction.
    corrected_phase, phase_model = _correct_phase_block(pols, phase_image, cellxy, results, telescope.magnification,
                                                        telescope.focus, telescope.surp_slope)
    # get RMSes before and after the fit
    in_rms = _compute_phase_rms_block(phase_image)
    out_rms = _compute_phase_rms_block(corrected_phase)
    #
    # # Convert output to convenient units
    results = _internal_to_external_parameters_block(results, wavelength, telescope, cellxy)
    errors  = _internal_to_external_parameters_block(np.sqrt(variances), wavelength, telescope, cellxy)
    #
    # return results, errors, corrected_phase, phase_model,
    return results, errors, corrected_phase, phase_model, in_rms, out_rms


def _internal_to_external_parameters(parameters, wavelength, telescope, cellxy):
    """
    Convert internal parameter array to convenient external units
    Args:
        parameters: Array in internal units
        wavelength: Observing wavelength, in meters
        telescope: Telescope object containing the optics parameters
        cellxy: Map cell spacing, in meters

    Returns:
        Array in convenient units, see _phase_fitting for more details
    """
    results = parameters
    # Convert to mm
    scaling = wavelength / 0.36
    results[3:] *= scaling
    # Sub-reflector tilt to degrees
    rad2deg = _convert_unit('rad', 'deg', 'trigonometric')
    results[6:8] *= rad2deg / (1000.0 * telescope.secondary_dist)
    # rescale phase ramp to pointing offset
    results[1:3] *= wavelength * rad2deg / 6. / cellxy
    return results * rad2deg


def _external_to_internal_parameters(exparameters, wavelength, telescope, cellxy):
    """
    Convert external parameter array to internal units
    Args:
        exparameters: Array in external units
        wavelength: Observing wavelength, in meters
        telescope: Telescope object containing the optics parameters
        cellxy: Map cell spacing, in meters

    Returns:
        Array in internal units, see _phase_fitting for more details
    """
    iNPARameters = exparameters
    # convert from mm
    scaling = wavelength / 0.36
    iNPARameters[3:] /= scaling
    # Sub-reflector tilt from degrees
    rad2deg = _convert_unit('rad', 'deg', 'trigonometric')
    iNPARameters[6:8] /= rad2deg / (1000.0 * telescope.secondary_dist)
    # rescale phase ramp to pointing offset
    iNPARameters[1:3] /= wavelength * rad2deg / 6. / cellxy
    iNPARameters /= rad2deg
    
    return iNPARameters


def _ignore_non_fitted(ignored, matrix, vector):
    """
    Disable the fitting of certain parameters by removing rows and columns from the design matrix and its associated
    vector
    Args:
        ignored: Array description of parameters to be ignored
        matrix: The design matrix
        vector: the vector associated with the design matrix

    Returns:
        The design matrix and its associated vector minus the rows and columns disabled
    """
    ndeleted = 0
    for ipar in range(NPAR):
        if ignored[ipar]:
            vector = np.delete(vector, ipar-ndeleted, 0)
            for axis in range(2):
                matrix = np.delete(matrix, ipar-ndeleted, axis)
            ndeleted += 1

    return matrix, vector


def _correct_phase(phase_image, cellxy, parameters, magnification, focal_length, phase_slope):
    """
    Corrects a phase image by using the phase model with the given parameters
    Args:
        phase_image: Grading phase map
        cellxy: Map cell spacing, in meters
        parameters: Parameters to be used in model determination
        magnification: Telescope Magnification
        focal_length: Nominal focal length, in meters
        phase_slope: Slope to apply to Q factor

    Returns:
        Corrected phase image and corresponfing phase_model
    """
    npix = phase_image.shape[0]
    ix0 = npix//2
    iy0 = npix//2
    phase_model = np.zeros((npix, npix))
    corrected_phase = np.zeros((npix, npix))
    phase_offset, x_pnt_off, y_pnt_off, x_focus_off, y_focus_off, z_focus_off, x_subref_tilt, y_subref_tilt, \
        x_cass_off, y_cass_off = parameters

    for iy in range(npix):
        for ix in range(npix):
            if not np.isnan(phase_image[ix, iy]):
                x_delta_pix = ix - ix0
                y_delta_pix = iy - iy0

                x_focus, y_focus, z_focus, x_tilt, y_tilt, x_cass, y_cass = _matrix_coeffs(x_delta_pix, y_delta_pix,
                                                                                           magnification, focal_length,
                                                                                           cellxy, phase_slope)
                corr = phase_offset + x_pnt_off * x_delta_pix + y_pnt_off * y_delta_pix + x_focus_off * x_focus
                corr += y_focus_off * y_focus + z_focus_off * z_focus + x_subref_tilt * x_tilt + y_subref_tilt * y_tilt
                corr += x_cass_off * x_cass + y_cass_off * y_cass
                corrected_phase[ix, iy] = phase_image[ix, iy] - corr
                phase_model[ix, iy] = corr

    return corrected_phase, phase_model


@njit(cache=False, nogil=True)
def _matrix_coeffs(x_delta_pix, y_delta_pix, magnification, focal_length, cellxy, phase_slope):
    """
    Computes the matrix coefficients used when building the design matrix and correcting the phase image
    Args:
        x_delta_pix: Distance from X reference pixel, in pixels
        y_delta_pix: Distance from Y reference pixel, in pixels
        magnification: Telescope Magnification
        focal_length: Nominal focal length
        cellxy: Map cell spacing, in meters
        phase_slope: Slope to apply to Q factor

    Returns:
        z_focus: Focus coefficient in Z direction
        x_foucs: Focus coefficient in X direction
        y_focus: Focus coefficient in Y direction
        x_tilt: Subreflector tilt coefficient in X direction
        y_tilt: Subreflector tilt coefficient in y direction
        x_cass: Cassegrain coefficient in x direction
        y_cass: Cassegrain coefficient in y direction
    """
    focal_path = focal_length / cellxy
    rad = np.sqrt(x_delta_pix * x_delta_pix + y_delta_pix * y_delta_pix)
    ang = np.arctan2(y_delta_pix, x_delta_pix)
    q_factor = rad / (2. * focal_path)
    q_factor_scaled = q_factor / magnification
    denominator = 1. + q_factor * q_factor
    denominator_scaled = 1. + q_factor_scaled * q_factor_scaled

    z_focus = (1. - q_factor * q_factor) / denominator + (1. - q_factor_scaled * q_factor_scaled) / denominator_scaled
    x_focus = -2. * np.cos(ang) * (
                q_factor / denominator - phase_slope * q_factor - q_factor_scaled / denominator_scaled)
    y_focus = -2. * np.sin(ang) * (
                q_factor / denominator - phase_slope * q_factor - q_factor_scaled / denominator_scaled)
    x_tilt = 2. * np.cos(ang) * (q_factor / denominator + q_factor / denominator_scaled)
    y_tilt = 2. * np.sin(ang) * (q_factor / denominator + q_factor / denominator_scaled)
    x_cass = -2. * np.cos(ang) * q_factor_scaled / denominator_scaled
    y_cass = -2. * np.sin(ang) * q_factor_scaled / denominator_scaled

    return x_focus, y_focus, z_focus, x_tilt, y_tilt, x_cass, y_cass


@njit(cache=False, nogil=True)
def _build_design_matrix_block(pols, xymin, xymax, cellxy, phase_image, amplitude_image, magnification, phase_slope,
                               focal_length):
    """
    Builds the design matrix to be used on the least squares fitting
    Args:
        pols: Indices of the polarizations to be used for phase fitting
        xymin: minimum of |x| and |y| used in correcting for pointing, focus, and feed offset. Negative values denote a
        range of SQRT(x*x + y*y)
        xymax: maximum of |x| and |y| used in correcting for pointing, focus, and feed offset. Negative values denote a
        range of SQRT(x*x + y*y)
        cellxy: Map cell spacing, in meters
        phase_image: Grading phase map
        amplitude_image: Grading amplitude map
        magnification: Telescope Magnification
        phase_slope: Slope to apply to Q factor
        focal_length: Nominal focal length, in meters

    Returns:
        Design matrix and associated vector
    """

    #   focal length in cellular units
    ntime = amplitude_image.shape[0]
    nchan = amplitude_image.shape[1]
    npix = phase_image.shape[3]
    npols = len(pols)
    ipol = 0
    ix0 = npix // 2
    iy0 = npix // 2

    matrix = np.zeros((ntime, nchan, npols, NPAR, NPAR))
    vector = np.zeros((ntime, nchan, npols, NPAR))
    ixymin = abs(xymin/cellxy)
    ixymax = abs(xymax/cellxy)
    min_squared_pix_radius = (xymin*xymin)/(cellxy*cellxy)
    max_squared_pix_radius = (xymax*xymax)/(cellxy*cellxy)
    
    for time in range(ntime):
        for chan in range(nchan):
            for pol in pols:
                for ix in range(npix):
                    x_delta_pix = abs(ix - ix0)
                    #   check absolute limits.
                    if xymin > 0.0 and x_delta_pix < ixymin:
                        continue
                    if xymax > 0.0 and x_delta_pix > ixymax:
                        continue
                    #   is this row of pixels outside
                    #   the outer ring?
                    if xymax < 0.0 and x_delta_pix * x_delta_pix > max_squared_pix_radius:
                        continue
                    for iy in range(npix):
                        #   ignore blanked pixels.
                        phase = phase_image[time, chan, pol, ix, iy]
                        if np.isnan(phase):
                            continue
                        #   check for inclusion.
                        y_delta_pix = abs(iy - iy0)
                        radius_pix_squared = x_delta_pix * x_delta_pix + y_delta_pix * y_delta_pix
                        #   inner limits.
                        if xymin > 0.0:
                            if y_delta_pix < ixymin:
                                continue
                        elif xymin < 0.0:
                            if radius_pix_squared < min_squared_pix_radius:
                                continue
                        #   outer limits.
                        if xymax > 0.0:
                            if y_delta_pix > ixymax:
                                continue
                        elif xymax < 0.0:
                            if radius_pix_squared > max_squared_pix_radius:
                                continue
                        
                        #   evaluate variables (in cells)
                        weight = amplitude_image[time, chan, pol, ix, iy]
                        
                        x_delta_pix = ix - ix0
                        y_delta_pix = iy - iy0
                        
                        x_focus, y_focus, z_focus, x_tilt, y_tilt, x_cass, y_cass = _matrix_coeffs(x_delta_pix, y_delta_pix,
                                                                                                   magnification, focal_length,
                                                                                                   cellxy, phase_slope)

                        #  build the design matrix.
                        vector[time, chan, ipol, 0] += phase * weight
                        vector[time, chan, ipol, 1] += phase * x_delta_pix * weight
                        vector[time, chan, ipol, 2] += phase * y_delta_pix * weight
                        vector[time, chan, ipol, 3] += phase * x_focus * weight
                        vector[time, chan, ipol, 4] += phase * y_focus * weight
                        vector[time, chan, ipol, 5] += phase * z_focus * weight
                        vector[time, chan, ipol, 6] += phase * x_tilt * weight
                        vector[time, chan, ipol, 7] += phase * y_tilt * weight
                        vector[time, chan, ipol, 8] += phase * x_cass * weight
                        vector[time, chan, ipol, 9] += phase * y_cass * weight
                        matrix[time, chan, ipol, 0, 0] += weight
                        matrix[time, chan, ipol, 0, 1] += x_delta_pix * weight
                        matrix[time, chan, ipol, 0, 2] += y_delta_pix * weight
                        matrix[time, chan, ipol, 0, 3] += x_focus * weight
                        matrix[time, chan, ipol, 0, 4] += y_focus * weight
                        matrix[time, chan, ipol, 0, 5] += z_focus * weight
                        matrix[time, chan, ipol, 0, 6] += x_tilt * weight
                        matrix[time, chan, ipol, 0, 7] += y_tilt * weight
                        matrix[time, chan, ipol, 0, 8] += x_cass * weight
                        matrix[time, chan, ipol, 0, 9] += y_cass * weight
                        matrix[time, chan, ipol, 1, 1] += x_delta_pix * x_delta_pix * weight
                        matrix[time, chan, ipol, 1, 2] += x_delta_pix * y_delta_pix * weight
                        matrix[time, chan, ipol, 1, 3] += x_delta_pix * x_focus * weight
                        matrix[time, chan, ipol, 1, 4] += x_delta_pix * y_focus * weight
                        matrix[time, chan, ipol, 1, 5] += x_delta_pix * z_focus * weight
                        matrix[time, chan, ipol, 1, 6] += x_delta_pix * x_tilt * weight
                        matrix[time, chan, ipol, 1, 7] += x_delta_pix * y_tilt * weight
                        matrix[time, chan, ipol, 1, 8] += x_delta_pix * x_cass * weight
                        matrix[time, chan, ipol, 1, 9] += x_delta_pix * y_cass * weight
                        matrix[time, chan, ipol, 2, 2] += y_delta_pix * y_delta_pix * weight
                        matrix[time, chan, ipol, 2, 3] += y_delta_pix * x_focus * weight
                        matrix[time, chan, ipol, 2, 4] += y_delta_pix * y_focus * weight
                        matrix[time, chan, ipol, 2, 5] += y_delta_pix * z_focus * weight
                        matrix[time, chan, ipol, 2, 6] += y_delta_pix * x_tilt * weight
                        matrix[time, chan, ipol, 2, 7] += y_delta_pix * y_tilt * weight
                        matrix[time, chan, ipol, 2, 8] += y_delta_pix * x_cass * weight
                        matrix[time, chan, ipol, 2, 9] += y_delta_pix * y_cass * weight
                        matrix[time, chan, ipol, 3, 3] += x_focus * x_focus * weight
                        matrix[time, chan, ipol, 3, 4] += x_focus * y_focus * weight
                        matrix[time, chan, ipol, 3, 5] += x_focus * z_focus * weight
                        matrix[time, chan, ipol, 3, 6] += x_focus * x_tilt * weight
                        matrix[time, chan, ipol, 3, 7] += x_focus * y_tilt * weight
                        matrix[time, chan, ipol, 3, 8] += x_focus * x_cass * weight
                        matrix[time, chan, ipol, 3, 9] += x_focus * y_cass * weight
                        matrix[time, chan, ipol, 4, 4] += y_focus * y_focus * weight
                        matrix[time, chan, ipol, 4, 5] += y_focus * z_focus * weight
                        matrix[time, chan, ipol, 4, 6] += y_focus * x_tilt * weight
                        matrix[time, chan, ipol, 4, 7] += y_focus * y_tilt * weight
                        matrix[time, chan, ipol, 4, 8] += y_focus * x_cass * weight
                        matrix[time, chan, ipol, 4, 9] += y_focus * y_cass * weight
                        matrix[time, chan, ipol, 5, 5] += z_focus * z_focus * weight
                        matrix[time, chan, ipol, 5, 6] += z_focus * x_tilt * weight
                        matrix[time, chan, ipol, 5, 7] += z_focus * y_tilt * weight
                        matrix[time, chan, ipol, 5, 8] += z_focus * x_cass * weight
                        matrix[time, chan, ipol, 5, 9] += z_focus * y_cass * weight
                        matrix[time, chan, ipol, 6, 6] += x_tilt * x_tilt * weight
                        matrix[time, chan, ipol, 6, 7] += x_tilt * y_tilt * weight
                        matrix[time, chan, ipol, 6, 8] += x_tilt * x_cass * weight
                        matrix[time, chan, ipol, 6, 9] += x_tilt * y_cass * weight
                        matrix[time, chan, ipol, 7, 7] += y_tilt * y_tilt * weight
                        matrix[time, chan, ipol, 7, 8] += y_tilt * x_cass * weight
                        matrix[time, chan, ipol, 7, 9] += y_tilt * y_cass * weight
                        matrix[time, chan, ipol, 8, 8] += x_cass * x_cass * weight
                        matrix[time, chan, ipol, 8, 9] += x_cass * y_cass * weight
                        matrix[time, chan, ipol, 9, 9] += y_cass * y_cass * weight

                ipol += 1

    return matrix, vector


def _reconstruct_full_results_block(results, variances, ignored):
    """
    Reconstruct the complete results and variances vectors from the ignored parameters
    Args:
        results: The output results from the least squares fit
        variances: The output variances from the least squares fit
        ignored: The array containing the information on which parameters were ignored

    Returns:
        reconstructed_results: full length result array, non fitted parameters replaced by zero
        reconstructed_variances: full length variance array, nan means unfitted parameter
    """
    ntime, nchan, npol = results.shape[0:3]
    reconstructed_results = np.zeros((ntime, nchan, npol, NPAR))
    reconstructed_variances = np.full((ntime, nchan, npol, NPAR), np.nan)
    for time in range(ntime):
        for chan in range(nchan):
            for pol in range(npol):
                jpar = 0
                for ipar in range(NPAR):
                    if not ignored[ipar]:
                        reconstructed_results[time, chan, pol, ipar] = results[time, chan, pol, jpar]
                        reconstructed_variances[time, chan, pol, ipar] = variances[time, chan, pol, jpar]
                        jpar += 1
    return reconstructed_results, reconstructed_variances


def _internal_to_external_parameters_block(parameters, wavelength, telescope, cellxy):
    """
    Convert internal parameter array to convenient external units
    Args:
        parameters: Array in internal units
        wavelength: Observing wavelength, in meters
        telescope: Telescope object containing the optics parameters
        cellxy: Map cell spacing, in meters

    Returns:
        Array in convenient units, see _phase_fitting for more details
    """
    ntime, nchan, npol = parameters.shape[:3]
    results = np.empty_like(parameters)
    for time in range(ntime):
        for chan in range(nchan):
            for pol in range(npol):
                results[time, chan, pol] = _internal_to_external_parameters(parameters[time, chan, pol],
                                                                            wavelength, telescope, cellxy)
    return results


def _ignore_non_fitted_block(ignored, matrix, vector):
    """
    Disable the fitting of certain parameters by removing rows and columns from the design matrix and its associated
    vector
    Args:
        ignored: Array description of parameters to be ignored
        matrix: The design matrix
        vector: the vector associated with the design matrix

    Returns:
        The design matrix and its associated vector minus the rows and columns disabled
    """

    newnpar = NPAR - int(round(np.sum(ignored)))
    if newnpar == NPAR:
        return matrix, vector

    else:
        ntime, nchan, npol = matrix.shape[:3]
        newmatrix = np.zeros((ntime, nchan, npol, newnpar, newnpar))
        newvector = np.zeros((ntime, nchan, npol, newnpar))
        for time in range(ntime):
            for chan in range(nchan):
                for pol in range(npol):
                    newmatrix[time, chan, pol], newvector[time, chan, pol] = \
                        _ignore_non_fitted(ignored, matrix[time, chan, pol], vector[time, chan, pol])
    return newmatrix, newvector


@njit(cache=False, nogil=True)
def _correct_phase_block(pols, phase_image, cellxy, parameters, magnification, focal_length, phase_slope):
    """
    Corrects a phase image by using the phase model with the given parameters
    Args:
        phase_image: Grading phase map
        cellxy: Map cell spacing, in meters
        parameters: Parameters to be used in model determination
        magnification: Telescope Magnification
        focal_length: Nominal focal length, in meters
        phase_slope: Slope to apply to Q factor

    Returns:
        Corrected phase image and corresponfing phase_model
    """
    npix = phase_image.shape[-1]
    ntime = phase_image.shape[0]
    nchan = phase_image.shape[1]
    ipol = 0
    ix0 = npix//2
    iy0 = npix//2
    phase_model = np.zeros_like(phase_image)
    corrected_phase = np.zeros_like(phase_image)

    for time in range(ntime):
        for chan in range(nchan):
            for pol in pols:
                phase_offset, x_pnt_off, y_pnt_off, x_focus_off, y_focus_off, z_focus_off, x_subref_tilt, \
                    y_subref_tilt, x_cass_off, y_cass_off = parameters[time, chan, ipol]
                for iy in range(npix):
                    for ix in range(npix):
                        phase = phase_image[time, chan, pol, ix, iy]
                        if not np.isnan(phase):

                            x_delta_pix = ix - ix0
                            y_delta_pix = iy - iy0

                            x_focus, y_focus, z_focus, x_tilt, y_tilt, x_cass, y_cass = \
                                _matrix_coeffs(x_delta_pix, y_delta_pix, magnification, focal_length, cellxy,
                                               phase_slope)
                            corr = phase_offset + x_pnt_off * x_delta_pix + y_pnt_off * y_delta_pix
                            corr += x_focus_off * x_focus + y_focus_off * y_focus + z_focus_off * z_focus
                            corr += x_subref_tilt * x_tilt + y_subref_tilt * y_tilt + x_cass_off * x_cass
                            corr += y_cass_off * y_cass
                            corrected_phase[time, chan, pol, ix, iy] = phase - corr
                            phase_model[time, chan, pol, ix, iy] = corr
                ipol += 1
    return corrected_phase, phase_model


def _build_ignored_array(pointing_offset, focus_xy_offsets, focus_z_offset, subreflector_tilt, cassegrain_offset):
    """

    Args:
        pointing_offset: Remove rows and columns related to pointing offsets
        focus_xy_offsets: Remove rows and columns related to XY focus offsets
        focus_z_offset: Remove the row and column related to Z focus offsets
        subreflector_tilt: Remove the rows and columns related to subreflector tilt
        cassegrain_offset: Remove the rows and columns related to cassegrain offsets

    Returns:
        Bool array contaning with True for the parameters to fitted and False for the rest
    """
    relevant = np.array([True, pointing_offset, pointing_offset, focus_xy_offsets, focus_xy_offsets, focus_z_offset,
                        subreflector_tilt, subreflector_tilt, cassegrain_offset, cassegrain_offset])
    return ~relevant


def _compute_phase_rms_block(phase_image):
    """
    Computes the RMS of the phase_image in a simple way
    Args:
        phase_image: Phase image to be analysed

    Returns:
        RMS of the phase_image
    """
    ntime, nchan, npol = phase_image.shape[:3]
    rms = np.zeros((ntime, nchan, npol))
    for time in range(ntime):
        for chan in range(nchan):
            for pol in range(npol):
                rms[time, chan, pol] = np.sqrt(np.nanmean(phase_image[time, chan, pol] ** 2))
    return rms

def _plot_antenna_chunk(parm_dict):
    antenna = parm_dict['this_antenna']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    plot_type = parm_dict['plot_type']
    plot_name = f'{destination}/{plot_type}_{antenna}_{ddi}.png'

    surface = parm_dict['panel_mds'].get_antenna(antenna, ddi)
    if plot_type == plot_types[0]:  # deviation plot
        surface.plot_surface(filename=plot_name, mask=False, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                             plotphase=False, unit=parm_dict['unit'])
    elif plot_type == plot_types[1]:  # phase plot
        surface.plot_surface(filename=plot_name, mask=False, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                             plotphase=True, unit=parm_dict['unit'])
    else:  # Ancillary plot
        surface.plot_surface(filename=plot_name, mask=True, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                             plotphase=False, unit=None)

