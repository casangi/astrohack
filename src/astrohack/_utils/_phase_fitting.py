import numpy as np
from astrohack._utils._linear_algebra import _least_squares_fit


npar = 10
i_x_pnt_off = 1
i_y_pnt_off = 2
i_x_focus_off = 3
i_y_focus_off = 4
i_z_focus_off = 5
i_x_subref_tilt = 6
i_y_subref_tilt = 7
i_x_cass_off = 8
i_y_cass_off = 9


def phase_fitting(wavelength, focal_length, xymin, xymax, cellxy, amplitude_image, phase_image,
                  disable_focus_xy_offsets, disable_focus_z_offset, disable_subreflector_tilt, disable_pointing_offset,
                  disable_cassegrain_offset, magnification, secondary_z_offset, phase_slope):
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
        wavelength: Observing wavelength, in meters
        focal_length: Nominal focal length, in meters
        xymin: minimum of |x| and |y| used in correcting for pointing, focus, and feed offset. Negative values denote a
        range of SQRT(x*x + y*y)
        xymax: maximum of |x| and |y| used in correcting for pointing, focus, and feed offset. Negative values denote a
        range of SQRT(x*x + y*y)
        cellxy: Map cell spacing, in meters
        amplitude_image: Grading amplitude map
        phase_image: Grading phase map
        disable_focus_xy_offsets: Disable subreflector offset model
        disable_focus_z_offset: Disable subreflector focus (z) model
        disable_subreflector_tilt: Enable subreflector rotation model
        disable_pointing_offset: Disable phase slope (pointing offset)
        disable_cassegrain_offset: Disable Cassegrain offsets (X, Y, Z)
        magnification: Telescope Magnification
        secondary_z_offset: Offset (prime focus to bottom subreflector)
        phase_slope: Slope to apply to Q factor

    Returns:
        results: Array containining the fit results in convenient units
        errors: Array containing the fit errors in convenient units
        corrected_phase: Phase map corrected for fitted parameters
        phase_model: Phase model used for the correction
        inrms: Phase RMS before fitting
        ourms: Phase RMS after fitting
    """

    matrix, vector = _build_design_matrix(xymin, xymax, cellxy, phase_image, amplitude_image, magnification,
                                          phase_slope, focal_length)
    if disable_pointing_offset:
        matrix, vector = _nullify(matrix, vector, i_x_pnt_off, i_y_pnt_off)
    if disable_focus_xy_offsets:
        matrix, vector = _nullify(matrix, vector, i_x_focus_off, i_y_focus_off)
    if disable_focus_z_offset:
        matrix, vector = _nullify(matrix, vector, i_z_focus_off, i_z_focus_off)
    if disable_subreflector_tilt:
        matrix, vector = _nullify(matrix, vector, i_x_subref_tilt, i_y_subref_tilt)
    if disable_cassegrain_offset:
        matrix, vector = _nullify(matrix, vector, i_x_cass_off, i_y_cass_off)

    #   compute the least squares solution.
    results, variances, residuals = _least_squares_fit(matrix, vector)
    #   apply the correction.
    corrected_phase, phase_model = _correct_phase(phase_image, cellxy, results, magnification, focal_length,
                                                  phase_slope)
    # get RMSes before and after the fit
    inrms = _compute_phase_rms(phase_image)
    ourms = _compute_phase_rms(corrected_phase)
    # Convert output to convenient units
    errors = np.sqrt(variances)
    # Convert to mm
    scaling = wavelength / 0.36
    results[3:] *= scaling
    errors[3:] *= scaling
    # Sub-reflector tilts to degrees
    rad2dg = np.pi / 180
    results[6:8] *= rad2dg / (190.0 * secondary_z_offset)
    errors[6:8] *= rad2dg / (190.0 * secondary_z_offset)
    # rescale phase slope to pointing offset
    results[1:3] *= wavelength / rad2dg / 6. / cellxy

    return results, errors, corrected_phase, phase_model, inrms, ourms


def _nullify(matrix, vector, start_par, end_par):
    """
    Nullify parameters that are not to be fitted
    Args:
        matrix: Fitting matrix
        vector: fitting vector
        start_par: First parameter to nullify
        end_par: Last parameter to nullify

    Returns:
        matrix and vector with nullified rows and columns
    """
    if start_par == end_par:
        loop = [start_par]
    else:
        loop = [start_par, end_par]
    for ipar in loop:
        vector[ipar] = 0.0
        for jpar in range(npar):
            matrix[jpar, ipar] = 0.0
            matrix[ipar, jpar] = 0.0

    return matrix, vector


def _build_design_matrix(xymin, xymax, cellxy, phase_image, amplitude_image, magnification, phase_slope, focal_length):
    """
    Builds the design matrix to be used on the least squares fitting
    Args:
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
    npix = phase_image.shape[0]
    #   focal length in cellular units
    ix0 = npix//2
    iy0 = npix//2
    matrix = np.zeros((npar, npar))
    vector = np.zeros(npar)
    ixymin = abs(xymin/cellxy)
    ixymax = abs(xymax/cellxy)
    min_squared_pix_radius = (xymin*xymin)/(cellxy*cellxy)
    max_squared_pix_radius = (xymax*xymax)/(cellxy*cellxy)

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
            if np.isnan(phase_image[ix, iy]):
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
            phase = phase_image[ix, iy]
            weight = amplitude_image[ix, iy]
            x_delta_pix = ix - ix0
            y_delta_pix = iy - iy0
            x_focus, y_focus, z_focus, x_tilt, y_tilt, x_cass, y_cass = _matrix_coeffs(x_delta_pix, y_delta_pix,
                                                                                       magnification, focal_length,
                                                                                       cellxy, phase_slope)
            #  build the design matrix.
            vector[0] += phase * weight
            vector[1] += phase * x_delta_pix * weight
            vector[2] += phase * y_delta_pix * weight
            vector[3] += phase * x_focus * weight
            vector[4] += phase * y_focus * weight
            vector[5] += phase * z_focus * weight
            vector[6] += phase * x_tilt * weight
            vector[7] += phase * y_tilt * weight
            vector[8] += phase * x_cass * weight
            vector[9] += phase * y_cass * weight
            matrix[0, 0] += weight
            matrix[0, 1] += x_delta_pix * weight
            matrix[0, 2] += y_delta_pix * weight
            matrix[0, 3] += x_focus * weight
            matrix[0, 4] += y_focus * weight
            matrix[0, 5] += z_focus * weight
            matrix[0, 6] += x_tilt * weight
            matrix[0, 7] += y_tilt * weight
            matrix[0, 8] += x_cass * weight
            matrix[0, 9] += y_cass * weight
            matrix[1, 1] += x_delta_pix * x_delta_pix * weight
            matrix[1, 2] += x_delta_pix * y_delta_pix * weight
            matrix[1, 3] += x_delta_pix * x_focus * weight
            matrix[1, 4] += x_delta_pix * y_focus * weight
            matrix[1, 5] += x_delta_pix * z_focus * weight
            matrix[1, 6] += x_delta_pix * x_tilt * weight
            matrix[1, 7] += x_delta_pix * y_tilt * weight
            matrix[1, 8] += x_delta_pix * x_cass * weight
            matrix[1, 9] += x_delta_pix * y_cass * weight
            matrix[2, 2] += y_delta_pix * y_delta_pix * weight
            matrix[2, 3] += y_delta_pix * x_focus * weight
            matrix[2, 4] += y_delta_pix * y_focus * weight
            matrix[2, 5] += y_delta_pix * z_focus * weight
            matrix[2, 6] += y_delta_pix * x_tilt * weight
            matrix[2, 7] += y_delta_pix * y_tilt * weight
            matrix[2, 8] += y_delta_pix * x_cass * weight
            matrix[2, 9] += y_delta_pix * y_cass * weight
            matrix[3, 3] += x_focus * x_focus * weight
            matrix[3, 4] += x_focus * y_focus * weight
            matrix[3, 5] += x_focus * z_focus * weight
            matrix[3, 6] += x_focus * x_tilt * weight
            matrix[3, 7] += x_focus * y_tilt * weight
            matrix[3, 8] += x_focus * x_cass * weight
            matrix[3, 9] += x_focus * y_cass * weight
            matrix[4, 4] += y_focus * y_focus * weight
            matrix[4, 5] += y_focus * z_focus * weight
            matrix[4, 6] += y_focus * x_tilt * weight
            matrix[4, 7] += y_focus * y_tilt * weight
            matrix[4, 8] += y_focus * x_cass * weight
            matrix[4, 9] += y_focus * y_cass * weight
            matrix[5, 5] += z_focus * z_focus * weight
            matrix[5, 6] += z_focus * x_tilt * weight
            matrix[5, 7] += z_focus * y_tilt * weight
            matrix[5, 8] += z_focus * x_cass * weight
            matrix[5, 9] += z_focus * y_cass * weight
            matrix[6, 6] += x_tilt * x_tilt * weight
            matrix[6, 7] += x_tilt * y_tilt * weight
            matrix[6, 8] += x_tilt * x_cass * weight
            matrix[6, 9] += x_tilt * y_cass * weight
            matrix[7, 7] += y_tilt * y_tilt * weight
            matrix[7, 8] += y_tilt * x_cass * weight
            matrix[7, 9] += y_tilt * y_cass * weight
            matrix[8, 8] += x_cass * x_cass * weight
            matrix[8, 9] += x_cass * y_cass * weight
            matrix[9, 9] += y_cass * y_cass * weight
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


def _compute_phase_rms(phase_image):
    """
    Computes the RMS of the phase_image in a simple way
    Args:
        phase_image: Phase image to be analysed

    Returns:
        RMS of the phase_image
    """
    return np.sqrt(np.nanmean(phase_image ** 2))
