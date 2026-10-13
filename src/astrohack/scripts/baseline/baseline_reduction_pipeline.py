import os
import time

import numpy as np
from astropy.utils import data as astropy_data

from astrohack import extract_locit, fringefit_locit, locit, open_position
from astrohack.utils.algorithms import data_statistics, rotate_to_gmt
from astrohack.utils.constants import clight, pol_str
from astrohack.utils.conversion import convert_unit
from astrohack.utils.pipeline_support import (
    MessageBoard,
    add_basic_info_and_parameters_to_report,
    asdm_test_and_import,
    base_name_determination,
    client_initialization,
    create_parser_with_base_options,
    initialization_check,
    list_input_tooltip,
    open_astrohack_file,
    parse_list_or_all,
    parse_list_or_none,
    proceed_check,
    run_astrohack_function,
    parse_list_or_all,
    base_name_determination,
    asdm_test_and_import,
    add_basic_info_and_parameters_to_report,
    create_parser_with_base_options,
    open_astrohack_file,
    client_initialization,
    parse_list_or_none,
    list_input_tooltip,
    parse_core_memory,
)
from astrohack.utils.text import (
    add_heading_to_html,
    add_preformatted_text_file_to_html,
    create_html_file_from_body,
    create_single_html_image_with_header,
    format_duration,
    lnbr,
    make_collapsible_block,
)
from astrohack.visualization.plot_tools import (
    close_figure,
    create_figure_and_axes,
    filter_duplicates_in_legend,
)

# This should hide the messages of downloading new IERS data by astropy
astropy_data.conf.show_progress = False


def parse(pipelie_type, stages):
    parser = create_parser_with_base_options(pipelie_type, stages)

    parser.add_argument(
        "-f",
        "--fringefit-source",
        default="0319+415",
        help="Fringe fit source, default is 0319+415",
    )

    parser.add_argument(
        "-i",
        "--intent",
        default="CALIBRATE_POINTING#ON_SOURCE",
        type=str,
        help="Intent for pointing observations.",
    )

    parser.add_argument(
        "-e",
        "--elevation-limit",
        type=float,
        default=10.0,
        help="Lowest elevation of data for consideration in degrees, default is %(default).1f",
    )

    parser.add_argument(
        "-p",
        "--polarization",
        type=str,
        choices=["both", "L", "R"],
        default="both",
        help="Which polarization hands to be used for locit processing, default is %(default)s",
    )

    parser.add_argument(
        "-c",
        "--combination",
        type=str,
        choices=["simple", "difference"],
        default="simple",
        help="How to combine different spws for locit processing, default is %(default)s",
    )

    parser.add_argument(
        "-k",
        "--fit-kterm",
        action="store_true",
        default=False,
        help="Fit antennas K term (i.e. Offset between azimuth and elevation axes)",
    )

    parser.add_argument(
        "-dr",
        "--fit-delay-rate",
        action="store_true",
        default=False,
        help="Fit delay rate",
    )

    parser.add_argument(
        "-l",
        "--delay-limits",
        type=float,
        default=0.1,
        help="Symmetrical limit for delay plots in nanoseconds, default is %(default)s which results in the limits being [-%(default)s, %(default)s]",
    )

    parser.add_argument(
        "--use-fringefit-locit",
        action="store_true",
        default=False,
        help="Use fringefit_locit to determine very large errors (order of  meters) in antenna positions (EXPERIMENTAL)",
    )

    parser.add_argument(
        "--exclude-scans",
        default=None,
        help="Select scans to exclude from locit delay fitting, "
        f"{list_input_tooltip('5,37')}, default is %(default)s",
    )

    return vars(parser.parse_args())


def fetch_ms_data(param_dict):
    import casatools

    # Ms data fetching and some consistency checks
    pnt_intent = "CALIBRATE_POINTING#ON_SOURCE"
    msmd = casatools.msmetadata()
    msmd.open(param_dict["msname"])
    ant_names = msmd.antennanames()
    field_names = msmd.fieldnames()
    spw_list = msmd.spwsforintent(pnt_intent)
    nchan = np.unique([msmd.nchan(i_spw) for i_spw in spw_list])
    msmd.done()

    param_dict["n_chan"] = nchan[0]

    error_msgs = []

    if param_dict["refant"] is None:
        error_msgs.append(
            "Reference antenna must be specified when starting from calibration"
        )
    else:
        if param_dict["refant"] not in ant_names:
            error_msgs.append(
                f"Chosen refant ({param_dict['refant']}) not present in ms."
            )
    if param_dict["fringefit_source"] not in field_names:
        error_msgs.append(
            f"Chosen fringefit source ({param_dict['fringefit_source']}) not present in ms."
        )
    if nchan.size != 1:
        error_msgs.append(
            "Spectral windows are not consistent with each other, is this really a pointing ms?"
        )
    if len(error_msgs) > 0:
        raise RuntimeError("\n".join(error_msgs))

    return param_dict


def param_init(param_dict: dict, msger: MessageBoard):

    base_name = base_name_determination(param_dict)
    param_dict = asdm_test_and_import(param_dict, base_name, msger)

    param_dict["pointing_only_ms"] = f"{base_name}.pnt.ms"
    param_dict["freq_averaged_ms"] = f"{base_name}.avg.ms"
    param_dict["fringefit_caltable"] = f"{base_name}.sbd"
    param_dict["phase_caltable"] = f"{base_name}.pha.gcal"
    param_dict["antpos_caltable"] = f"{base_name}.antpos"
    param_dict["locit_name"] = f"{base_name}.locit.zarr"
    param_dict["position_name"] = f"{base_name}.position.zarr"
    param_dict["exports_name"] = f"{base_name}.exports"
    param_dict["report_name"] = f"{base_name}-report.html"
    param_dict["parallel"] = param_dict["ncores"] > 1
    param_dict["memory_per_core"] = parse_core_memory(param_dict["memory_per_core"])

    param_dict["antenna"] = parse_list_or_all(param_dict, "antenna")
    param_dict["spw"] = parse_list_or_all(param_dict, "spw", list_type=int)
    param_dict["delay_limits"] = [
        -param_dict["delay_limits"],
        param_dict["delay_limits"],
    ]
    param_dict["exclude_scans"] = parse_list_or_none(param_dict, "exclude_scans")

    if param_dict["starting_stage"] == "calibration":
        param_dict = fetch_ms_data(param_dict)

    initialization_check(param_dict, "Baseline determination parameters")
    if param_dict["use_fringefit_locit"] and param_dict["antenna"] == "all":
        proceed_check(
            param_dict,
            "Fringefit using all antennas and all fields may take a LONG time, proceed?",
        )

    return param_dict


def run_casa_pre_locit_steps(param_dict: dict, msger: MessageBoard):
    run_casatask(
        "split",
        {
            "vis": param_dict["msname"],
            "outputvis": param_dict["pointing_only_ms"],
            "intent": param_dict["intent"],
            "datacolumn": "data",
        },
        msger,
        intended_output=param_dict["pointing_only_ms"],
        overwrite=param_dict["overwrite"],
    )

    if param_dict["use_fringefit_locit"]:
        if param_dict["antenna"] == "all":
            antenna_str = ""
        else:
            antenna_str = ",".join(param_dict["antenna"])

        run_casatask(
            "fringefit",
            {
                "vis": param_dict["pointing_only_ms"],
                "caltable": param_dict["fringefit_caltable"],
                "solint": "inf",
                "refant": param_dict["refant"],
                "antenna": antenna_str,
                "minsnr": 3.0,
                "zerorates": True,
                "globalsolve": True,
                "niter": 100,
            },
            msger,
            intended_output=param_dict["fringefit_caltable"],
            overwrite=param_dict["overwrite"],
        )

    else:
        fringefit_was_run = run_casatask(
            "fringefit",
            {
                "vis": param_dict["pointing_only_ms"],
                "caltable": param_dict["fringefit_caltable"],
                "field": param_dict["fringefit_source"],
                "solint": "inf",
                "refant": param_dict["refant"],
                "minsnr": 3.0,
                "zerorates": True,
                "globalsolve": True,
                "niter": 100,
            },
            msger,
            intended_output=param_dict["fringefit_caltable"],
            overwrite=param_dict["overwrite"],
        )
        if fringefit_was_run:
            run_casatask(
                "applycal",
                {
                    "vis": param_dict["pointing_only_ms"],
                    "gaintable": [param_dict["fringefit_caltable"]],
                    "interp": ["nearest"],
                    "parang": False,
                },
                msger,
            )

        # Now we create a new dataset that is colapsed on the channel axis
        # within each spw, also create a flagversion to store current flag
        # state on the averaged MS
        freq_average_was_run = run_casatask(
            "split",
            {
                "vis": param_dict["pointing_only_ms"],
                "outputvis": param_dict["freq_averaged_ms"],
                "datacolumn": "corrected",
                "keepflags": False,  #
                "width": param_dict["n_chan"],
            },
            msger,
            intended_output=param_dict["freq_averaged_ms"],
            overwrite=param_dict["overwrite"],
        )
        if freq_average_was_run:
            run_casatask(
                "flagmanager",
                {
                    "vis": param_dict["freq_averaged_ms"],
                    "mode": "save",
                    "versionname": "original",
                },
                msger,
            )

        gaincal_was_run = run_casatask(
            "gaincal",
            {
                "vis": param_dict["freq_averaged_ms"],
                "caltable": param_dict["phase_caltable"],
                "solint": "10min",
                "refant": param_dict["refant"],
                "refantmode": "strict",  # Maybe we should use strict for this application?
                "minblperant": 3,
                "minsnr": 3.0,
                "gaintype": "G",  # G is for gain
                "calmode": "p",  # p is for phase
                "solmode": "L1",  # -> least squares
            },
            msger,
            intended_output=param_dict["phase_caltable"],
            overwrite=param_dict["overwrite"],
        )
        if gaincal_was_run:
            run_casatask(
                "applycal",
                {
                    "vis": param_dict["freq_averaged_ms"],
                    "gaintable": [param_dict["phase_caltable"]],
                    "interp": ["nearest"],
                    "parang": False,
                },
                msger,
            )

        if not param_dict["assume_yes"]:
            run_casatask(
                "plotms",
                {
                    "vis": param_dict["freq_averaged_ms"],
                    "xaxis": "time",
                    "yaxis": "phase",
                    "ydatacolumn": "corrected",
                    "field": "*",
                    "avgtime": "10",
                    "correlation": "RR,LL",
                    "coloraxis": "spw",
                    "antenna": param_dict["refant"],
                    "iteraxis": "baseline",
                },
                msger,
            )
            proceed_check(param_dict, "Are phase solutions reasonable?")

    return


def run_astrohack_locit(param_dict: dict, msger: MessageBoard):
    astrohack_param_dict = {
        "fringefit_caltable": param_dict["fringefit_caltable"],
        "cal_table": param_dict["phase_caltable"],
        "locit_name": param_dict["locit_name"],
        "position_name": param_dict["position_name"],
        "ant": param_dict["antenna"],
        "ddi": param_dict["spw"],
        "overwrite": param_dict["overwrite"],
        "fit_kterm": param_dict["fit_kterm"],
        "fit_delay_rate": param_dict["fit_delay_rate"],
        "elevation_limit": param_dict["elevation_limit"],
        "polarization": param_dict["polarization"],
        "combine_ddis": param_dict["combination"],
        "exclude_scans": param_dict["exclude_scans"],
        "parallel": param_dict["parallel"],
    }
    if param_dict["use_fringefit_locit"]:
        locit_functions = [fringefit_locit]
    else:
        locit_functions = [extract_locit, locit]
    for function in locit_functions:
        status, exec_exception = run_astrohack_function(
            astrohack_param_dict, function, msger
        )
        if not status:
            raise RuntimeError(
                f"{function.__name__} failed see above for details."
            ) from exec_exception


def run_astrohack_exports(param_dict: dict, msger: MessageBoard):
    astrohack_param_dict = {
        "destination": param_dict["exports_name"],
        "ant": param_dict["antenna"],
        "filename": f"{param_dict['exports_name']}/parminator.par",
        "parallel": False,
        "dpi": param_dict["dpi"],
        "delay_limits": param_dict["delay_limits"],
    }
    position_mds = open_astrohack_file(open_position, param_dict["position_name"])
    plotting_methods = [
        position_mds.plot_array_configuration,
        position_mds.plot_source_positions,
        position_mds.plot_delays,
        position_mds.plot_position_corrections,
        position_mds.export_locit_fit_results,
        position_mds.export_results_to_parminator,
    ]

    for plot_method in plotting_methods:
        status, exec_exception = run_astrohack_function(
            astrohack_param_dict, plot_method, msger
        )
        if not status:
            raise RuntimeError(
                f"{plot_method.__name__} failed see above for details."
            ) from exec_exception

    position_mds.print_source_table(
        save_to=f"{param_dict['exports_name']}/source_table.txt"
    )
    position_mds.print_array_configuration(
        save_to=f"{param_dict['exports_name']}/array_configuration.txt"
    )
    return


def make_single_antenna_phase_over_time(
    exports_name,
    refant_name,
    ant_name,
    corr_axis,
    scan_list,
    plt_times,
    raw_phases,
    corr_phases,
    dpi,
):
    import matplotlib.ticker as ticker

    def forward(x_val):
        return np.interp(x_val, plt_times, scan_list)

    def inverse(x_val):
        return np.interp(x_val, scan_list, plt_times)

    corr_styles = ["+", "x"]
    fig, ax = create_figure_and_axes(None, [1, 1])
    for i_corr, corr in enumerate(corr_axis):
        legend = f"{corr}"
        ax.plot(
            plt_times,
            raw_phases[:, :, i_corr],
            f"r{corr_styles[i_corr]}",
            label=f"DATA, {legend}",
        )
        ax.plot(
            plt_times,
            corr_phases[:, :, i_corr],
            f"b{corr_styles[i_corr]}",
            label=f"CORRECTED, {legend}",
        )
    raw_rms = data_statistics(raw_phases)["rms"]
    corr_rms = data_statistics(corr_phases)["rms"]
    ax.legend()
    ax.set_xlabel("Time since observation start [hour]")
    ax.set_ylabel("Phase [deg]")
    ax.set_ylim([-180, 180])
    secax = ax.secondary_xaxis("top", functions=(forward, inverse))
    secax.set_xlabel("Scan number")
    secax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    if corr_rms < 0.3 * raw_rms:
        status = "Greatly improved!"
    elif corr_rms < 0.8 * raw_rms:
        status = "Improved!"
    else:
        status = "Not improved..."
    ax.set_title(
        f"data RMS: {raw_rms:.2f} [deg], corrected RMS: {corr_rms:.2f} [deg] ({status})"
    )
    filter_duplicates_in_legend(ax)
    close_figure(
        fig,
        f"Phases before and after antenna position corrections for {ant_name}&{refant_name}",
        f"{exports_name}/phases-antpos-{ant_name}.png",
        dpi,
        False,
        True,
    )
    return


def make_all_antenna_phases_plot(
    param_dict: dict, ant_names: list, msger: MessageBoard
):
    from casacoretables import tables

    start = time.time()
    msger.one_liner("Preparing phase over time plots...")

    rad2deg = convert_unit("rad", "deg", "trigonometric")
    sec2hr = convert_unit("sec", "hour", "time")

    avg_ms = param_dict["freq_averaged_ms"]
    ant_table = tables.table(avg_ms + "/ANTENNA")
    ms_ant_names = ant_table.getcol("NAME")
    ant_table.close()

    pol_table = tables.table(avg_ms + "/POLARIZATION")
    corr_axis = pol_str[pol_table.getcol("CORR_TYPE")[0, :]]
    pol_table.close()
    corr_select = np.logical_or(corr_axis == "RR", corr_axis == "LL")
    corr_axis = corr_axis[corr_select]

    ms_table = tables.table(avg_ms)
    vis_time = ms_table.getcol("TIME")
    vis_raw = ms_table.getcol("DATA")[:, 0, corr_select]
    vis_corr = ms_table.getcol("CORRECTED_DATA")[:, 0, corr_select]
    ant_1 = ms_table.getcol("ANTENNA1")
    ant_2 = ms_table.getcol("ANTENNA2")
    scans = ms_table.getcol("SCAN_NUMBER")
    weights = ms_table.getcol("WEIGHT")[:, corr_select]
    flags = ms_table.getcol("FLAG")[:, 0, corr_select]
    spws = ms_table.getcol("DATA_DESC_ID")
    ms_table.close()

    spw_list = np.unique(spws)
    scan_list = np.unique(scans)
    n_spws = spw_list.shape[0]
    n_scans = scan_list.shape[0]
    n_corrs = corr_axis.shape[0]

    first_time = np.min(np.unique(vis_time))

    refant_id = ms_ant_names.index(param_dict["refant"])
    for ant_name in ant_names:
        if ant_name == param_dict["refant"]:
            continue
        ant_id = ms_ant_names.index(ant_name)
        ant_select = np.logical_or(
            np.logical_and(ant_1 == ant_id, ant_2 == refant_id),
            np.logical_and(ant_2 == ant_id, ant_1 == refant_id),
        )
        ant_sel_scans = scans[ant_select]
        ant_sel_flags = flags[ant_select]
        ant_sel_time = vis_time[ant_select]

        ant_sel_raw = np.ma.masked_array(vis_raw[ant_select], ant_sel_flags)
        ant_sel_corr = np.ma.masked_array(vis_corr[ant_select], ant_sel_flags)
        ant_sel_weights = weights[ant_select]
        ant_sel_spws = spws[ant_select]

        phase_shape = (n_scans, n_spws, n_corrs)
        plt_times = np.full_like(scan_list, np.nan, dtype=float)
        raw_phases = np.full(phase_shape, np.nan, dtype=float)
        corr_phases = np.full(phase_shape, np.nan, dtype=float)
        for i_scan, scan in enumerate(scan_list):
            scan_select = ant_sel_scans == scan
            plt_times[i_scan] = (
                np.average(ant_sel_time[scan_select]) - first_time
            ) * sec2hr
            scan_sel_weights = ant_sel_weights[scan_select]
            scan_sel_corr = ant_sel_corr[scan_select]
            scan_sel_raw = ant_sel_raw[scan_select]
            scan_sel_spws = ant_sel_spws[scan_select]
            for i_spw, spw in enumerate(spw_list):
                spw_select = scan_sel_spws == spw
                avg_raw = np.ma.average(
                    scan_sel_raw[spw_select],
                    weights=scan_sel_weights[spw_select],
                    axis=0,
                )
                raw_phases[i_scan, i_spw, :] = np.angle(avg_raw) * rad2deg
                avg_corr = np.ma.average(
                    scan_sel_corr[spw_select],
                    weights=scan_sel_weights[spw_select],
                    axis=0,
                )
                corr_phases[i_scan, i_spw, :] = np.angle(avg_corr) * rad2deg

        make_single_antenna_phase_over_time(
            param_dict["exports_name"],
            param_dict["refant"],
            ant_name,
            corr_axis,
            scan_list,
            plt_times,
            raw_phases,
            corr_phases,
            param_dict["dpi"],
        )

    stop = time.time()
    msger.one_liner(f"Plotting finished in {format_duration(stop - start)}")
    return


def run_post_locit_plots(param_dict: dict, msger: MessageBoard):
    pos_corrections = []
    ant_names = []
    position_mds = open_position(param_dict["position_name"])
    for ant_key, ant_xdt in position_mds.items():
        attributes = ant_xdt.attrs
        geo_delays, _ = rotate_to_gmt(
            np.copy(attributes["position_fit"]),
            attributes["position_error"],
            attributes["antenna_info"]["longitude"],
        )
        pos_corrections.extend(clight * geo_delays)
        ant_names.append(attributes["antenna_info"]["name"])

    try:
        import casatasks  # noqa: F401

        param_dict["casatasks_found"] = True
    except ModuleNotFoundError:
        param_dict["casatasks_found"] = False
        msger.one_liner(
            "casatasks not available, skipping evaluation of phases after antenna position corrections..."
        )

    if param_dict["casatasks_found"]:
        gencal_was_run = run_casatask(
            "gencal",
            {
                "vis": param_dict["freq_averaged_ms"],
                "caltable": param_dict["antpos_caltable"],
                "caltype": "antpos",
                "antenna": ",".join(ant_names),
                "parameter": pos_corrections,
            },
            msger,
            intended_output=param_dict["antpos_caltable"],
            overwrite=param_dict["overwrite"],
        )
        if gencal_was_run:
            run_casatask(
                "applycal",
                {
                    "vis": param_dict["freq_averaged_ms"],
                    "gaintable": [param_dict["antpos_caltable"]],
                    "interp": ["nearest"],
                    "parang": False,
                },
                msger,
            )

        make_all_antenna_phases_plot(param_dict, ant_names, msger)
    return


def prepare_html_report(param_dict: dict, msger: MessageBoard):
    msger.one_liner("Preparing report...")
    start = time.time()
    exports_name = param_dict["exports_name"]
    pos_mds = open_position(param_dict["position_name"])
    combination_word = pos_mds.root.attrs["combine_specifier"]
    del pos_mds

    images_to_include = {
        "locit_source_table_fk5.png": [
            "Source positions over the sky",
            f"{param_dict['exports_name']}/source_table.txt",
        ],
        "locit_array_configuration.png": [
            "VLA configuration during observation",
            f"{param_dict['exports_name']}/array_configuration.txt",
        ],
        f"position_corrections_combined_{combination_word}.png": [
            "Graphical representation of antenna position corrections",
            None,
        ],
    }

    report_title = f"Baseline Report for {param_dict['filename']}"
    html_body = add_heading_to_html(report_title, 1)

    html_body += add_basic_info_and_parameters_to_report(param_dict)

    for image_file, image_contents in images_to_include.items():
        image_desc = image_contents[0]
        accompanying_table = image_contents[1]
        image_path = f"{exports_name}/{image_file}"
        html_body += (
            f"{create_single_html_image_with_header(image_path, image_desc)}{lnbr}"
        )
        if accompanying_table is not None:
            html_body += add_preformatted_text_file_to_html(
                accompanying_table,
                "",
            )

    html_body += add_preformatted_text_file_to_html(
        f"{exports_name}/position_combined_{combination_word}_fit_results.txt",
        "Measured antenna position corrections",
    )

    html_body += add_preformatted_text_file_to_html(
        f"{exports_name}/parminator.par",
        "Proposed parminator corrections",
    )

    pos_mds = open_position(param_dict["position_name"])
    antenna_name_list = [key.split("_")[1] for key in pos_mds.keys()]
    del pos_mds
    for ant_name in antenna_name_list:
        if ant_name != param_dict["refant"]:
            delay_plot_file = f"{exports_name}/position_delays_ant_{ant_name}_combined_{combination_word}.png"
            if os.path.exists(delay_plot_file):
                ant_html = create_single_html_image_with_header(
                    delay_plot_file,
                    "Measured and fitted delays",
                    heading_level=3,
                )
                if (not param_dict["use_fringefit_locit"]) and param_dict[
                    "casatasks_found"
                ]:
                    ant_html += create_single_html_image_with_header(
                        f"{exports_name}/phases-antpos-{ant_name}.png",
                        "Phase before and after correction",
                        heading_level=3,
                    )
            else:
                ant_html = add_heading_to_html(
                    f"Fit has not succeeded for {ant_name}!", 1
                )
            html_body += make_collapsible_block(
                ant_html,
                add_heading_to_html(f"Results for {ant_name}", 2),
                f"ant_{ant_name}",
            )
    create_html_file_from_body(html_body, report_title, param_dict["report_name"])
    stop = time.time()
    msger.one_liner(f"Report finished in {stop - start:.2f} seconds")


def main():
    pipeline_type = "baseline"
    pipeline_start = time.time()
    msger = MessageBoard()
    print()
    stages = ["calibration", "locit", "exports", "report"]
    msger.welcome_message(pipeline_type)

    param_dict = param_init(parse(pipeline_type, stages), msger)
    processing_stage = param_dict["starting_stage"]

    if processing_stage == stages[0]:
        run_casa_pre_locit_steps(param_dict, msger)
        processing_stage = stages[1]

    param_dict["processing_stage"] = processing_stage
    client = client_initialization(param_dict, stages[1:3])

    if processing_stage == stages[1]:
        run_astrohack_locit(param_dict, msger)
        processing_stage = stages[2]

    if processing_stage == stages[2]:
        run_astrohack_exports(param_dict, msger)
        if not param_dict["use_fringefit_locit"]:
            run_post_locit_plots(param_dict, msger)
        processing_stage = stages[3]

    if client is not None:
        client.shutdown()

    if processing_stage == stages[3]:
        prepare_html_report(param_dict, msger)

    pipeline_end = time.time()
    msger.goodbye_message(pipeline_type, param_dict, pipeline_end - pipeline_start)
    return
