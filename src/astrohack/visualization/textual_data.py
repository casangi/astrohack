import numpy as np

from astrohack.core.image_comparison_tool import extract_rms_from_xds
from astrohack.utils import rad_to_deg_str, twopi, fixed_format_error, dynamic_format
from astrohack.antenna import Telescope, AntennaSurface
from astrohack.utils import (
    convert_unit,
    clight,
    notavail,
    param_to_list,
    add_prefix,
    format_value_error,
    rotate_to_gmt,
    format_frequency,
    format_wavelength,
    format_value_unit,
    length_units,
    trigo_units,
    format_label,
    create_pretty_table,
    string_to_ascii_file,
    compute_antenna_relative_off,
)
import toolviper.utils.logger as logger

from astrohack.utils.phase_fitting import aips_par_names
from astrohack.utils.tools import get_telescope_lat_lon_rad


def export_to_parminator(data_dict, parm_dict):
    combined = parm_dict["combined"]

    kterm_present = data_dict._meta_data["fit_kterm"]

    full_antenna_list = Telescope(data_dict._meta_data["telescope_name"]).ant_list
    selected_antenna_list = param_to_list(parm_dict["ant"], data_dict, "ant")
    threshold = parm_dict["correction_threshold"]

    parmstr = ""
    for ant_name in full_antenna_list:
        ant_key = add_prefix(ant_name, "ant")

        if ant_key in selected_antenna_list:
            if ant_key in data_dict.keys():
                if combined:
                    antenna = data_dict[ant_key]
                else:
                    antenna = data_dict[ant_key][f'ddi_{parm_dict["ddi"]}']

                parmstr += _export_parminator_antenna(
                    antenna.attrs, threshold, kterm_present
                )

    string_to_ascii_file(parmstr, parm_dict["filename"])


def _export_parminator_antenna(attributes, threshold, kterm_present):

    axes = ["X", "Y", "Z"]
    delays, _ = rotate_to_gmt(
        np.copy(attributes["position_fit"]),
        attributes["position_error"],
        attributes["antenna_info"]["longitude"],
    )
    station = attributes["antenna_info"]["station"]

    outstr = ""
    for iaxis, delay in enumerate(delays):
        correction = delay * clight
        if np.abs(correction) > threshold:
            outstr += f"{station}, ,{axes[iaxis]},${correction: .4f}\n"

    if kterm_present:
        correction = attributes["koff_fit"] * clight
        if np.abs(correction) > threshold:
            outstr += f"{station}, ,K,${correction: .4f}\n"
    return outstr


def export_locit_fit_results(data_dict, parm_dict):
    """
    Export fit results to a txt file listing the different DDIs as different solutions if data is not combined
    Args:
        data_dict: the mds content
        parm_dict: Dictionary of the parameters given to the calling function

    Returns:
    text file with fit results in convenient units
    """
    pos_unit = parm_dict["position_unit"]
    del_unit = parm_dict["delay_unit"]
    pha_unit = parm_dict["phase_unit"]
    len_fact = convert_unit("m", pos_unit, "length")
    del_fact = convert_unit("sec", del_unit, kind="time")
    pha_fact = convert_unit("rad", pha_unit, kind="trigonometric")
    pos_fact = len_fact * clight
    combined = parm_dict["combined"]

    if combined:
        field_names = [
            "Antenna",
            "Station",
            f"RMS [{del_unit}]",
            f"RMS [{pha_unit}]",
            f"F. delay [{del_unit}]",
            f"X offset [{pos_unit}]",
            f"Y offset [{pos_unit}]",
            f"Z offset [{pos_unit}]",
        ]
        specifier = "combined_" + data_dict._meta_data["combine_ddis"]
    else:
        field_names = [
            "Antenna",
            "Station",
            "DDI",
            f"RMS [{del_unit}]",
            f"RMS [{pha_unit}]",
            f"F. delay [{del_unit}]",
            f"X offset [{pos_unit}]",
            f"Y offset [{pos_unit}]",
            f"Z offset [{pos_unit}]",
        ]
        specifier = "separated_ddis"
    kterm_present = data_dict._meta_data["fit_kterm"]
    rate_present = data_dict._meta_data["fit_delay_rate"]
    if kterm_present:
        field_names.extend([f"K offset [{pos_unit}]"])
    if rate_present:
        tim_unit = parm_dict["time_unit"]
        slo_unit = f"{del_unit}/{tim_unit}"
        slo_fact = del_fact / convert_unit("day", tim_unit, "time")
        field_names.extend([f"Rate [{slo_unit}]"])
    else:
        slo_unit = notavail
        slo_fact = 1.0

    table = create_pretty_table(field_names)
    full_antenna_list = Telescope(data_dict._meta_data["telescope_name"]).ant_list
    selected_antenna_list = param_to_list(parm_dict["ant"], data_dict, "ant")

    for ant_name in full_antenna_list:
        ant_key = add_prefix(ant_name, "ant")
        if ant_name == data_dict._meta_data["reference_antenna"]:
            ant_name += " (ref)"

        if ant_key in selected_antenna_list:
            if ant_key in data_dict.keys():
                antenna = data_dict[ant_key]
                if combined:
                    row = [ant_name, antenna.attrs["antenna_info"]["station"]]
                    table.add_row(
                        _export_locit_xds(
                            row,
                            antenna.attrs,
                            del_fact,
                            pha_fact,
                            pos_fact,
                            slo_fact,
                            pos_unit,
                            del_unit,
                            kterm_present,
                            rate_present,
                        )
                    )
                else:
                    ddi_list = param_to_list(
                        parm_dict["ddi"], data_dict[ant_key], "ddi"
                    )
                    for ddi_key in ddi_list:
                        row = [
                            ant_name,
                            antenna[ddi_key].attrs["antenna_info"]["station"],
                            ddi_key.split("_")[1],
                        ]
                        table.add_row(
                            _export_locit_xds(
                                row,
                                antenna[ddi_key].attrs,
                                del_fact,
                                pha_fact,
                                pos_fact,
                                slo_fact,
                                pos_unit,
                                del_unit,
                                kterm_present,
                                rate_present,
                            )
                        )

    print(table.get_string())
    string_to_ascii_file(
        table.get_string(),
        parm_dict["destination"] + f"/position_{specifier}_fit_results.txt",
    )


def _export_locit_xds(
    row,
    attributes,
    del_fact,
    pha_fact,
    pos_fact,
    slo_fact,
    pos_unit,
    del_unit,
    kterm_present,
    rate_present,
):
    """
    Export the data from a single X array DataSet attributes to a table row (a list)
    Args:
        row: row onto which the data results are to be added
        attributes: The XDS attributes dictionary
        del_fact: Delay unit scaling factor
        pos_fact: Position unit scaling factor
        slo_fact: Delay rate unit scaling factor
        kterm_present: Is the elevation axis offset term present?
        rate_present: Is the delay rate term present?

    Returns:
    The filled table row
    """

    delay_rms = np.sqrt(attributes["chi_squared"])
    mean_freq = np.nanmean(attributes["frequency"])
    phase_rms = twopi * mean_freq * delay_rms
    row.append(f"{delay_rms*del_fact:4.2e}")
    row.append(f"{phase_rms*pha_fact:5.1f}")

    sig_scale_pos = convert_unit("mm", pos_unit, "length")
    sig_scale_del = 1e-3 * convert_unit("nsec", del_unit, "time")

    row.append(
        fixed_format_error(
            attributes["fixed_delay_fit"],
            attributes["fixed_delay_error"],
            del_fact,
            sig_scale_del,
        )
    )
    position, poserr = rotate_to_gmt(
        np.copy(attributes["position_fit"]),
        attributes["position_error"],
        attributes["antenna_info"]["longitude"],
    )

    for i_pos in range(3):
        row.append(
            fixed_format_error(position[i_pos], poserr[i_pos], pos_fact, sig_scale_pos)
        )
    if kterm_present:
        row.append(
            fixed_format_error(
                attributes["koff_fit"],
                attributes["koff_error"],
                pos_fact,
                sig_scale_pos,
            )
        )
    if rate_present:
        row.append(
            fixed_format_error(
                attributes["rate_fit"],
                attributes["rate_error"],
                slo_fact,
                sig_scale_del,
            )
        )
    return row


def export_screws_chunk(parm_dict):
    """
    Chunk function for the user facing function export_screws
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    export_name = parm_dict["destination"] + f"/panel_screws_{antenna}_{ddi}."
    xds = parm_dict["xds_data"]
    telescope = Telescope(xds.attrs["telescope_name"])
    surface = AntennaSurface(xds, telescope, reread=True)
    surface.export_screws(export_name + "txt", unit=parm_dict["unit"])
    surface.plot_screw_adjustments(export_name + "png", parm_dict)


def export_gains_table_chunk(parm_dict):
    in_waves = parm_dict["wavelengths"]
    in_freqs = parm_dict["frequencies"]
    ant = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    xds = parm_dict["xds_data"]
    telescope = Telescope.from_xds(xds)
    antenna = AntennaSurface(xds, telescope, reread=True)
    frequency = clight / antenna.wavelength

    if in_waves is None and in_freqs is None:
        try:
            wavelengths = telescope.gain_wavelengths
        except AttributeError:
            msg = f"Telescope {telescope.name} has no predefined list of wavelengths to compute gains"
            logger.error(msg)
            logger.info("Please provide one in the arguments")
            raise Exception(msg)
    else:
        wave_fac = convert_unit(parm_dict["wavelength_unit"], "m", "length")
        freq_fac = convert_unit(parm_dict["frequency_unit"], "Hz", "frequency")
        wavelengths = []
        if in_waves is not None:
            if isinstance(in_waves, float) or isinstance(in_waves, int):
                in_waves = [in_waves]
            for in_wave in in_waves:
                wavelengths.append(wave_fac * in_wave)
        if in_freqs is not None:
            if isinstance(in_freqs, float) or isinstance(in_freqs, int):
                in_freqs = [in_freqs]
            for in_freq in in_freqs:
                wavelengths.append(clight / freq_fac / in_freq)

    db = "dB"
    rmsunit = parm_dict["rms_unit"]
    rmses = antenna.get_rms(rmsunit)

    field_names = [
        "Frequency",
        "Wavelength",
        "Before panel",
        "After panel",
        "Theoretical Max.",
    ]
    table = create_pretty_table(field_names)

    outstr = f'# Gain estimates for {telescope.name} antenna {ant.split("_")[1]}\n'
    outstr += f"# Based on a measurement at {format_frequency(frequency)}, {format_wavelength(antenna.wavelength)}\n"
    outstr += f"# Antenna surface RMS before adjustment: {format_value_unit(rmses[0], rmsunit)}\n"
    outstr += f"# Antenna surface RMS after adjustment: {format_value_unit(rmses[1], rmsunit)}\n"
    outstr += 1 * "\n"

    for wavelength in wavelengths:
        prior, theo = antenna.gain_at_wavelength(False, wavelength)
        after, _ = antenna.gain_at_wavelength(True, wavelength)
        row = [
            format_frequency(clight / wavelength),
            format_wavelength(wavelength),
            format_value_unit(prior, db),
            format_value_unit(after, db),
            format_value_unit(theo, db),
        ]
        table.add_row(row)

    outstr += table.get_string()
    string_to_ascii_file(
        outstr, parm_dict["destination"] + f"/panel_gains_{ant}_{ddi}.txt"
    )


def export_phase_fit_chunk(parm_dict):
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    phase_fit_results = parm_dict["xds_data"].attrs["phase_fitting"]
    angle_unit = parm_dict["angle_unit"]
    length_unit = parm_dict["length_unit"]
    field_names = ["Parameter", "Value", "Unit"]
    alignment = ["l", "r", "c"]
    outstr = ""

    for mapkey, map_dict in phase_fit_results.items():
        for freq, freq_dict in map_dict.items():
            for pol, pol_dict in freq_dict.items():
                outstr += f'* {mapkey.replace("_", " ")}, Frequency {format_frequency(freq)}, polarization state {pol}:\n\n '
                table = create_pretty_table(field_names, alignment)
                for par_name in aips_par_names:
                    item = pol_dict[par_name]
                    val = item["value"]
                    err = item["error"]
                    unit = item["unit"]
                    if unit in length_units:
                        fac = convert_unit(unit, length_unit, "length")
                    elif unit in trigo_units:
                        fac = convert_unit(unit, angle_unit, "trigonometric")
                    else:
                        msg = f"Unknown unit {unit}"
                        logger.error(msg)
                        raise Exception(msg)

                    row = [
                        format_label(par_name),
                        format_value_error(fac * val, fac * err, 1.0, 1e-4),
                        unit,
                    ]
                    table.add_row(row)

                outstr += table.get_string() + "\n\n"

    string_to_ascii_file(outstr, f"{destination}/image_phase_fit_{antenna}_{ddi}.txt")


def export_zernike_fit_chunk(parm_dict):
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    zernike_coeffs = parm_dict["xds_data"]["ZERNIKE_COEFFICIENTS"].values
    rms = parm_dict["xds_data"]["ZERNIKE_FIT_RMS"].values
    corr_axis = parm_dict["xds_data"].orig_pol.values
    freq_axis = parm_dict["xds_data"].chan.values
    ntime = zernike_coeffs.shape[0]
    osa_indices = parm_dict["xds_data"].osa.values
    destination = parm_dict["destination"]

    field_names = ["Indices", "Real", "Imaginary"]
    alignment = ["l", "c", "c"]
    outstr = ""

    for itime in range(ntime):
        for ichan, freq in enumerate(freq_axis):
            for icorr, corr in enumerate(corr_axis):
                outstr += f"* map {itime}, Frequency {format_frequency(freq)}, Correlation {corr}:\n"
                outstr += f"   Fit RMS = {rms[itime, ichan, icorr].real:.8f} + {rms[itime, ichan, icorr].imag:.8f}*i\n\n"
                table = create_pretty_table(field_names, alignment)
                for icoeff, coeff in enumerate(zernike_coeffs[itime, ichan, icorr]):
                    row = [
                        osa_indices[icoeff],
                        f"{coeff.real:.8f}",
                        f"{coeff.imag:.8f}",
                    ]
                    table.add_row(row)

                outstr += table.get_string() + "\n\n"

    string_to_ascii_file(outstr, f"{destination}/image_zernike_fit_{antenna}_{ddi}.txt")


def print_array_configuration(params, ant_dict, telescope_name):
    """Backend for printing the array configuration onto a table

    Args:
        params: Parameter dictionary crafted by the calling function
        ant_dict: Parameter dictionary crafted by the calling function
        telescope_name: Name of the telescope used in observations
    """
    telescope = Telescope(telescope_name)
    relative = params["relative"]

    print(f"\n{telescope_name} antennas, # of antennas {len(ant_dict.keys())}:")
    if relative:
        nfields = 5
        field_names = [
            "Name",
            "Station",
            "East [m]",
            "North [m]",
            "Elevation [m]",
            "Distance [m]",
        ]
        tel_lon, tel_lat, tel_rad = get_telescope_lat_lon_rad(telescope)
    else:
        nfields = 4
        field_names = ["Name", "Station", "Longitude", "Latitude", "Radius [m]"]

    table = create_pretty_table(field_names)
    for ant_name in telescope.ant_list:
        ant_key = "ant_" + ant_name
        if ant_key in ant_dict:
            antenna = ant_dict[ant_key]
            if antenna["reference"]:
                ant_name += " (ref)"
            row = [ant_name, antenna["station"]]
            if relative:
                offsets = compute_antenna_relative_off(
                    antenna, tel_lon, tel_lat, tel_rad
                )
                row.extend(
                    [
                        f"{offsets[0]:.4f}",
                        f"{offsets[1]:.4f}",
                        f"{offsets[2]:.4f}",
                        f"{offsets[3]:.4f}",
                    ]
                )
            else:
                row.extend(
                    [
                        rad_to_deg_str(antenna["longitude"]),
                        rad_to_deg_str(antenna["latitude"]),
                        f'{antenna["radius"]:.4f}',
                    ]
                )
            table.add_row(row)
        else:
            row = [ant_name]
            for i_field in range(nfields):
                row.append(notavail)
            table.add_row(row)

    print(table)
    return


def create_fits_comparison_rms_table(parameters, xdt):
    image_list = xdt.children
    rms_unit = parameters["rms_unit"]

    fields = [
        "Image",
        "Reference",
        f"Original RMS [{rms_unit}]",
        f"Resampled RMS [{rms_unit}]",
        f"Reference RMS [{rms_unit}]",
        f"Residuals RMS [{rms_unit}]",
    ]

    factor = convert_unit("m", rms_unit, "length")

    table = create_pretty_table(fields)
    for image in image_list:

        image_xds = xdt[image]["Image"].to_dataset()
        reference_xds = xdt[image]["Reference"].to_dataset()

        img_rms_dict = extract_rms_from_xds(image_xds)
        ref_rms_dict = extract_rms_from_xds(reference_xds)
        values = np.array(
            [
                img_rms_dict["original"],
                img_rms_dict["resampled"],
                ref_rms_dict["original"],
                img_rms_dict["residuals"],
            ]
        )
        values *= factor

        row = [image_xds.attrs["filename"], reference_xds.attrs["filename"]]
        for val in values:
            row.append(f"{val:{dynamic_format(val)}}")

        table.add_row(row)

    outstr = f'RMS comparison table from {parameters["zarr_data_tree"]}:\n'
    outstr += table.get_string()
    string_to_ascii_file(outstr, parameters["table_file"])
    if parameters["print_table"]:
        print(table)
    return
