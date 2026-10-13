import time

from astrohack import (
    beamcut,
    extract_holog,
    extract_pointing,
    open_beamcut,
    open_pointing,
)
from astrohack.utils.pipeline_support import (
    MessageBoard,
    add_basic_info_and_parameters_to_report,
    basic_holography_parser,
    client_initialization,
    common_parameter_initialization_for_holography,
    initialization_check,
    open_astrohack_file,
    run_astrohack_function,
    run_casatask,
)
from astrohack.utils.text import (
    add_heading_to_html,
    add_preformatted_text_file_to_html,
    create_html_file_from_body,
    create_single_html_image_with_header,
    lnbr,
    make_collapsible_block,
)


def parse(pipeline_type, stages):
    parser = basic_holography_parser(pipeline_type, stages)

    # beam cut only options will come here

    return vars(parser.parse_args())


def param_init(pipeline_type: str, msger: MessageBoard):
    extensions = {
        "delay_cal": ".dcal",
        "bandpass_cal": ".bcal",
        "gain_cal": ".gcal",
        "point": ".point.zarr",
        "holog": ".holog.zarr",
        "beamcut": ".beamcut.zarr",
        "exports": ".exports",
        "report": "-report.html",
    }

    stages = [
        "calibration",
        "extract_pointing",
        "extract_holog",
        "beamcut",
        "exports",
        "report",
    ]

    param_dict = common_parameter_initialization_for_holography(
        pipeline_type, stages, extensions, parse, msger
    )

    # Extra parameter initialization comes here

    initialization_check(param_dict, "Beam cut reduction parameters")
    return param_dict, stages


def run_casa_calibration(param_dict, msger):
    gaintable = []
    delay_ok = run_casatask(
        "gaincal",
        {
            "vis": param_dict["msname"],
            "caltable": param_dict["delay_cal_name"],
            "refant": param_dict["refant"],
            "solint": "inf",
            "spw": param_dict["quacked_spw_selection"],
            "scan": param_dict["calibration_scans"],
            "gaintype": "K",
        },
        msger,
        intended_output=param_dict["delay_cal_name"],
        overwrite=param_dict["overwrite"],
    )

    if delay_ok:
        gaintable.append(param_dict["delay_cal_name"])
        bandpass_ok = run_casatask(
            "bandpass",
            {
                "vis": param_dict["msname"],
                "caltable": param_dict["bandpass_cal_name"],
                "refant": param_dict["refant"],
                "solint": "10s",
                "spw": param_dict["quacked_spw_selection"],
                "scan": param_dict["calibration_scans"],
                "solnorm": True,
                "gaintable": gaintable,
            },
            msger,
            intended_output=param_dict["bandpass_cal_name"],
            overwrite=param_dict["overwrite"],
        )
    else:
        bandpass_ok = False

    if bandpass_ok:
        gaintable.append(param_dict["bandpass_cal_name"])
        gaincal_ok = run_casatask(
            "gaincal",
            {
                "vis": param_dict["msname"],
                "caltable": param_dict["gain_cal_name"],
                "refant": param_dict["refant"],
                "calmode": "ap",
                "solint": "inf",
                "spw": param_dict["quacked_spw_selection"],
                "minsnr": 2,
                "minblperant": 2,
                "scan": param_dict["calibration_scans"],
                "gaintable": gaintable,
            },
            msger,
            intended_output=param_dict["gain_cal_name"],
            overwrite=param_dict["overwrite"],
        )
    else:
        gaincal_ok = False

    if gaincal_ok:
        gaintable.append(param_dict["gain_cal_name"])
        run_casatask(
            "applycal",
            {
                "vis": param_dict["msname"],
                "field": f"{param_dict['beamcut_field']}",
                "spw": param_dict["quacked_spw_selection"],
                "applymode": "calonly",
                "gaintable": gaintable,
            },
            msger,
        )

    return


def run_astrohack_reduction(param_dict, msger):
    # Astrohack convenience changes
    param_dict["ant"] = param_dict["antenna"]
    param_dict["ddi"] = param_dict["spw"]
    param_dict["exclude_antennas"] = param_dict["exclude_bad_antennas"]

    param_dict["ms_name"] = param_dict["msname"]

    status = True
    exec_exception = None
    exec_list = [
        ["extract_holog", extract_pointing],
        ["beamcut", extract_holog],
        ["exports", beamcut],
    ]
    for next_stage, function in exec_list:
        if status and param_dict["processing_stage"] == function.__name__:
            status, exec_exception = run_astrohack_function(
                param_dict,
                function,
                msger,
            )
            if status:
                param_dict["processing_stage"] = next_stage

    if not status:
        raise RuntimeError(
            f"{param_dict['processing_stage']} failed, see above for details."
        ) from exec_exception

    return


def run_astrohack_exports(param_dict, msger):
    param_dict["destination"] = param_dict["exports_name"]
    pnt_mds = open_astrohack_file(open_pointing, param_dict["point_name"])
    bmc_mds = open_astrohack_file(open_beamcut, param_dict["beamcut_name"])

    plotting_methods = [
        pnt_mds.plot_array_configuration,
        bmc_mds.plot_in_amplitude,
        bmc_mds.plot_in_phase,
        bmc_mds.plot_in_tapering,
        bmc_mds.plot_lm_offsets,
        bmc_mds.export_report,
    ]
    if param_dict["plot_pointing"]:
        param_dict["plot_antennas_separately"] = True
        plotting_methods.append(pnt_mds.plot_pointing_in_time)

    for plot_method in plotting_methods:
        status, exec_exception = run_astrohack_function(param_dict, plot_method, msger)
        if not status:
            raise RuntimeError(
                f"{plot_method.__name__} failed see above for details."
            ) from exec_exception

    return


def prepare_html_report(param_dict, msger):
    msger.one_liner("Preparing report...")
    start = time.time()
    exports_name = param_dict["exports_name"]
    report_title = f"Beamcut report for {param_dict['filename']}"

    html_body = add_heading_to_html(report_title, 1)
    html_body += add_basic_info_and_parameters_to_report(param_dict)
    html_body += create_single_html_image_with_header(
        f"{exports_name}/point_array_configuration.png",
        "Array configuration during observation",
        heading_level=2,
    )
    html_body += f"{lnbr}<br>{lnbr}"

    bmc_mds = open_beamcut(param_dict["beamcut_name"])
    if bmc_mds is None:
        raise RuntimeError(f"{param_dict['beamcut_name']} not found")
    antenna_list = [ant_key.split("_")[-1] for ant_key in bmc_mds.keys()]
    ddi_list = [
        ddi_key.split("_")[-1] for ddi_key in bmc_mds[f"ant_{antenna_list[0]}"].keys()
    ]
    for ant_name in antenna_list:
        ant_html = ""
        if param_dict["plot_pointing"]:
            ant_html += create_single_html_image_with_header(
                f"{exports_name}/point_directional_cosines_ant_{ant_name}.png",
                "Pointing over time:",
                heading_level=3,
            )
        ant_html += create_single_html_image_with_header(
            f"{exports_name}/beamcut_lm_offsets_ant_{ant_name}_ddi_{ddi_list[0]}.png",
            "Pointing over sky:",
            heading_level=3,
        )
        for ddi_name in ddi_list:
            spw_html = create_single_html_image_with_header(
                f"{exports_name}/beamcut_db_ant_{ant_name}_ddi_{ddi_name}.png",
                "Beam cut in dB",
                heading_level=4,
            )
            spw_html += create_single_html_image_with_header(
                f"{exports_name}/beamcut_amplitude_ant_{ant_name}_ddi_{ddi_name}.png",
                "Beam cut in amplitude",
                heading_level=4,
            )
            spw_html += create_single_html_image_with_header(
                f"{exports_name}/beamcut_phase_ant_{ant_name}_ddi_{ddi_name}.png",
                "Beam cut in phase",
                heading_level=4,
            )
            spw_html += add_preformatted_text_file_to_html(
                f"{exports_name}/beamcut_report_ant_{ant_name}_ddi_{ddi_name}.txt",
                "Beam cut fit report",
                heading_level=4,
            )
            ant_html += make_collapsible_block(
                spw_html,
                add_heading_to_html(f"\t{ant_name} spectral window {ddi_name}:", 3),
                f"ant_{ant_name}_spw_{ddi_name}",
            )
        # collapsible wrapping here
        html_body += make_collapsible_block(
            ant_html,
            add_heading_to_html(f"Beam cut data for {ant_name}:", 2),
            f"ant_{ant_name}",
        )

    create_html_file_from_body(html_body, report_title, param_dict["report_name"])
    stop = time.time()
    msger.one_liner(f"Report finished in {stop - start:.2f} seconds")
    return


def main():
    pipeline_type = "beamcut"
    pipeline_start = time.time()
    msger = MessageBoard()
    print()
    msger.welcome_message(pipeline_type)

    main_param_dict, stages = param_init(pipeline_type, msger)

    astrohack_stages = stages[1:5]
    main_param_dict["processing_stage"] = main_param_dict["starting_stage"]

    if main_param_dict["processing_stage"] == stages[0]:
        run_casa_calibration(main_param_dict, msger)
        main_param_dict["processing_stage"] = astrohack_stages[0]

    client = client_initialization(main_param_dict, astrohack_stages)

    if main_param_dict["processing_stage"] in astrohack_stages[:-1]:
        run_astrohack_reduction(main_param_dict, msger)

    if main_param_dict["processing_stage"] == "exports":
        run_astrohack_exports(main_param_dict, msger)
        main_param_dict["processing_stage"] = "report"

    if main_param_dict["processing_stage"] == "report":
        prepare_html_report(main_param_dict, msger)

    if client is not None:
        client.shutdown()

    pipeline_end = time.time()
    msger.goodbye_message(pipeline_type, main_param_dict, pipeline_end - pipeline_start)
