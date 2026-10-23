import time
from pathlib import Path

import numpy as np

from astrohack import (
    combine,
    extract_holog,
    extract_pointing,
    holog,
    open_image,
    open_panel,
    open_pointing,
    panel,
)
from astrohack.antenna.panel_fitting import PANEL_MODEL_DICT
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


def parse(pipeline_type: str, stages: list):
    parser = basic_holography_parser(pipeline_type, stages)

    # Extra holography options come here
    parser.add_argument(
        "--grid-size",
        default=None,
        type=int,
        help="Choose grid size for beam image in pixels, the default is to guess it from the data.",
    )

    parser.add_argument(
        "--cell-size",
        default=None,
        type=float,
        help="Choose cell size for beam image in radians, the default is to guess it from the data.",
    )

    parser.add_argument(
        "--grid-interpolation-mode",
        default="gaussian",
        type=str,
        help="Choose interpolation mode for beam image (default is %(default)s)",
    )

    parser.add_argument(
        "--padding-factor",
        default=10,
        type=int,
        help="Padding factor applied to the beam image before the FFT to produce the apertures (default is %(default)s)",
    )

    parser.add_argument(
        "--zernike-n-order",
        default=4,
        type=int,
        help="Zernike N polynomial order to be fitted to the apertures (default is %(default)s)",
    )

    parser.add_argument(
        "-z",
        "--use-zernike-phase-fitting",
        action="store_true",
        help=(
            "Use zernike phase fitting instead of regular perturbation phase fitting "
            "(ngVLA uses this even when option is not given)."
        ),
    )

    parser.add_argument(
        "-c",
        "--combine",
        action="store_true",
        help="Combine SPWs to improve SNR (all selected SPWs will be combined)",
    )

    parser.add_argument(
        "--clip-type",
        default="sigma",
        type=str,
        choices=["sigma", "none", "absolute", "relative", "noise_threshold"],
        help="Choose the type of clipping algorithm to use in the apertures before fitting panels (default is %(default)s)",
    )

    parser.add_argument(
        "--clip-level",
        default=3,
        type=float,
        help="Choose the level for the chosen clipping algorithm, panel help for more details (default is %(default)s)",
    )

    parser.add_argument(
        "--panel-model",
        default="flexible",
        type=str,
        choices=list(PANEL_MODEL_DICT.keys()),
        help="Choose the panel model to use (default is %(default)s)",
    )

    parser.add_argument(
        "--panel-margins",
        default=0.05,
        type=float,
        help="Define the fraction of the edge of the panels to be excluded from fitting (default is %(default)s)",
    )

    parser.add_argument(
        "-u",
        "--screw-unit",
        default="mils",
        help="Unit to present screw adjustments (default is %(default)s)",
    )

    return vars(parser.parse_args())


def param_init(pipeline_type: str, msger: MessageBoard):
    extensions = {
        "delay_cal": ".dcal",
        "bandpass_cal": ".bcal",
        "gain_cal": ".gcal",
        "point": ".point.zarr",
        "holog": ".holog.zarr",
        "image": ".image.zarr",
        "combine": ".combine.zarr",
        "panel": ".panel.zarr",
        "exports": ".exports",
        "report": "-report.html",
    }
    stages = [
        "calibration",
        "extract_pointing",
        "extract_holog",
        "holog",
        "panel",
        "exports",
        "report",
    ]
    param_dict = common_parameter_initialization_for_holography(
        pipeline_type, stages, extensions, parse, msger, for_holograpy=True
    )

    # extra parameter initialization comes here

    initialization_check(
        param_dict, f"{pipeline_type.capitalize()} reduction parameters"
    )
    return param_dict, stages


def run_casa_calibration(param_dict: dict, msger: MessageBoard):
    delay_success = []
    intended_delay_out = param_dict["delay_cal_name"]
    overwrite_delay = param_dict["overwrite"]
    append_delay = False
    for base_band in param_dict["quacked_base_band_selection"]:
        delay_success.append(
            run_casatask(
                "gaincal",
                {
                    "vis": param_dict["msname"],
                    "caltable": param_dict["delay_cal_name"],
                    "refant": param_dict["refant"],
                    "solint": "10s",
                    "spw": base_band,
                    "scan": param_dict["calibration_scans"],
                    "minblperant": 3,
                    "gaintype": "K",
                    "append": append_delay,
                    "calmode": "p",
                    "combine": "spw",
                },
                msger,
                intended_output=intended_delay_out,
                overwrite=overwrite_delay,
            )
        )
        append_delay = True
        overwrite_delay = False
        intended_delay_out = None

    gaintable = []
    spwmap = []
    delay_ok = np.all(delay_success)

    if delay_ok:
        gaintable.append(param_dict["delay_cal_name"])
        spwmap.append(param_dict["delay_spwmap"])
        bandpass_ok = run_casatask(
            "bandpass",
            {
                "vis": param_dict["msname"],
                "caltable": param_dict["bandpass_cal_name"],
                "refant": param_dict["refant"],
                "solint": "inf",
                "spw": param_dict["quacked_spw_selection"],
                "scan": param_dict["calibration_scans"],
                "solnorm": True,
                "gaintable": gaintable,
                "spwmap": spwmap,
            },
            msger,
            intended_output=param_dict["bandpass_cal_name"],
            overwrite=param_dict["overwrite"],
        )
    else:
        bandpass_ok = False

    if bandpass_ok:
        gaintable.append(param_dict["bandpass_cal_name"])
        spwmap.append(param_dict["full_spwmap"])
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
                "spwmap": spwmap,
            },
            msger,
            intended_output=param_dict["gain_cal_name"],
            overwrite=param_dict["overwrite"],
        )
    else:
        gaincal_ok = False

    if gaincal_ok:
        gaintable.append(param_dict["gain_cal_name"])
        spwmap.append(param_dict["full_spwmap"])
        run_casatask(
            "applycal",
            {
                "vis": param_dict["msname"],
                "field": f"{param_dict['holography_field']}",
                "spw": param_dict["quacked_spw_selection"],
                "applymode": "calonly",
                "gaintable": gaintable,
                "spwmap": spwmap,
            },
            msger,
        )

    return


def run_astrohack_reduction(param_dict: dict, msger: MessageBoard):
    # Astrohack convenience changes
    param_dict["ant"] = param_dict["antenna"]
    param_dict["ddi"] = param_dict["spw"]
    param_dict["exclude_antennas"] = param_dict["exclude_bad_antennas"]
    param_dict["ms_name"] = param_dict["msname"]
    if param_dict["use_zernike_phase_fitting"]:
        param_dict["phase_fit_engine"] = "zernike"
    else:
        param_dict["phase_fit_engine"] = "perturbations"
    bckp_img_name = param_dict["image_name"]

    status = True
    exec_exception = None

    exec_list = [
        ["extract_holog", extract_pointing],
        ["holog", extract_holog],
        ["panel", holog],
        ["exports", panel],
    ]

    for next_stage, function in exec_list:
        if status and param_dict["processing_stage"] == function.__name__:
            if param_dict["processing_stage"] == "panel" and param_dict["combine"]:
                param_dict["image_name"] = param_dict["combine_name"]

            status, exec_exception = run_astrohack_function(
                param_dict,
                function,
                msger,
            )
            if param_dict["processing_stage"] == "holog" and param_dict["combine"]:
                status, exec_exception = run_astrohack_function(
                    param_dict,
                    combine,
                    msger,
                )

            if status:
                param_dict["processing_stage"] = next_stage
    param_dict["image_name"] = bckp_img_name
    if not status:
        raise RuntimeError(
            f"{param_dict['processing_stage']} failed, see above for details."
        ) from exec_exception

    return


def run_astrohack_exports(param_dict: dict, msger: MessageBoard):
    param_dict["destination"] = param_dict["exports_name"]
    # This key forces pnl_mds.plot_antennas to do all aperture plots
    param_dict["plot_type"] = "all"
    pnt_mds = open_astrohack_file(open_pointing, param_dict["point_name"])
    # Should we include any holog plots? also what to do in case of combine? combine philosophy seems to be wrong...
    # hlg_mds = open_astrohack_file(open_holog, param_dict["holog_name"])
    pnl_mds = open_astrohack_file(open_panel, param_dict["panel_name"])

    # This is a patch to the image_name in the case there was a combine so that plots are combined.
    first_ddi_key = list(next(iter(pnl_mds.values()), None).keys())[0]
    if first_ddi_key == "ddi_99":
        img_mds = open_astrohack_file(open_image, param_dict["combine_name"])
    else:
        img_mds = open_astrohack_file(open_image, param_dict["image_name"])

    export_methods = [
        pnt_mds.plot_array_configuration,
        img_mds.plot_beams,
        img_mds.export_phase_fit_results,
        img_mds.export_zernike_fit_results,
        img_mds.plot_zernike_model,
        pnl_mds.plot_antennas,
        pnl_mds.export_gain_tables,
    ]
    if param_dict["plot_pointing"]:
        param_dict["plot_antennas_separately"] = True
        export_methods.append(pnt_mds.plot_pointing_in_time)

    for export_method in export_methods:
        status, exec_exception = run_astrohack_function(
            param_dict, export_method, msger
        )
        if not status:
            raise RuntimeError(
                f"{export_method.__name__} failed see above for details."
            ) from exec_exception

    # export_screws is called separately to avoid changing units in other plotting routines.œ
    param_dict["unit"] = param_dict["screw_unit"]
    status, exec_exception = run_astrohack_function(
        param_dict, pnl_mds.export_screws, msger
    )
    if not status:
        raise RuntimeError(
            f"{pnl_mds.export_screws.__name__} failed see above for details."
        ) from exec_exception
    return


def prepare_html_report(param_dict: dict, msger: MessageBoard):
    msger.one_liner("Preparing report...")
    start = time.time()
    exports_name = param_dict["exports_name"]
    report_title = f"Holography report for {param_dict['filename']}"
    html_body = add_heading_to_html(report_title, 1)
    html_body += add_basic_info_and_parameters_to_report(param_dict)
    html_body += create_single_html_image_with_header(
        f"{exports_name}/point_array_configuration.png",
        "Array configuration during observation",
        heading_level=2,
    )
    html_body += f"{lnbr}<br>{lnbr}"

    pnl_mds = open_astrohack_file(open_panel, param_dict["panel_name"])
    for ant_key, ant_xds in pnl_mds.items():
        ant_name = ant_key.split("_")[1]
        ant_html = ""
        if param_dict["plot_pointing"]:
            ant_html += create_single_html_image_with_header(
                f"{exports_name}/point_directional_cosines_{ant_key}.png",
                "Pointing over time:",
                heading_level=3,
            )

        for ddi_key in ant_xds.keys():
            ddi_name = ddi_key.split("_")[1]
            # summary to be done here per antenna/ddi
            summ_name = f"{exports_name}/obs_summary_{ant_key}_{ddi_key}.txt"
            pnl_mds.observation_summary(
                summary_file=summ_name,
                ant=ant_name,
                ddi=ddi_name,
                parallel=False,
                print_summary=False,
            )

            spw_html = add_preformatted_text_file_to_html(
                summ_name, "Observation Summary", 3
            )
            phase_fit_file = f"{exports_name}/image_phase_fit_{ant_key}_{ddi_key}.txt"
            if Path(phase_fit_file).is_file():
                spw_html += add_preformatted_text_file_to_html(
                    phase_fit_file,
                    "Phase fitting results",
                    3,
                )
            spw_html += add_preformatted_text_file_to_html(
                f"{exports_name}/panel_gains_{ant_key}_{ddi_key}.txt",
                "Predicted antenna gains",
                3,
            )

            ### Beam part
            beam_block = ""
            for stokes_par in ["I", "Q", "U", "V"]:
                beam_block += create_single_html_image_with_header(
                    f"{exports_name}/image_beam_polar_{ant_key}_{ddi_key}_pol_{stokes_par}.png",
                    f"Beam for Stokes {stokes_par}",
                    heading_level=3,
                )

            spw_html += make_collapsible_block(
                beam_block,
                "Beam plots",
                f"{ant_key}_spw_{ddi_name}_beam",
            )

            ### Zernike part
            zernike_block = add_preformatted_text_file_to_html(
                f"{exports_name}/image_zernike_fit_{ant_key}_{ddi_key}.txt",
                "Zernike fit results",
                heading_level=3,
            )
            for correlation in ["RR", "RL", "LR", "LL"]:
                zernike_block += create_single_html_image_with_header(
                    f"{exports_name}/image_zernike_model_{ant_key}_{ddi_key}_corr_{correlation}.png",
                    f"Zernike model for {correlation} correlation",
                    heading_level=3,
                )

            spw_html += make_collapsible_block(
                zernike_block,
                "Zernike fitting results",
                f"{ant_key}_spw_{ddi_name}_zernike",
            )

            ### Aperture part
            aperture_block = ""
            for plot_type in [
                "amplitude",
                "mask",
                "phase_original",
                "deviation_original",
                "deviation_correction",
                "deviation_residual",
            ]:
                aperture_block += create_single_html_image_with_header(
                    f"{exports_name}/panel_{plot_type}_{ant_key}_{ddi_key}.png",
                    f"Aperture {plot_type.capitalize().replace('_', ' ')}",
                    heading_level=3,
                )

            spw_html += make_collapsible_block(
                aperture_block,
                "Aperture plots",
                f"{ant_key}_spw_{ddi_name}_aperture",
            )

            ### Screw part
            screw_block = create_single_html_image_with_header(
                f"{exports_name}/panel_screws_{ant_key}_{ddi_key}.png",
                "Screw adjustment map",
                heading_level=3,
            )
            screw_block += add_preformatted_text_file_to_html(
                f"{exports_name}/panel_screws_{ant_key}_{ddi_key}.txt",
                "Full list of screw corrections",
                heading_level=3,
            )

            spw_html += make_collapsible_block(
                screw_block,
                "Proposed screw adjustments",
                f"{ant_key}_spw_{ddi_name}_screws",
            )

            ant_html += make_collapsible_block(
                spw_html,
                add_heading_to_html(f"\t{ant_name} spectral window {ddi_name}:", 3),
                f"{ant_key}_spw_{ddi_name}",
            )

        html_body += make_collapsible_block(
            ant_html,
            add_heading_to_html(f"Holography data for {ant_name}:", 2),
            ant_key,
        )

    create_html_file_from_body(html_body, report_title, param_dict["report_name"])
    stop = time.time()
    msger.one_liner(f"Report finished in {stop - start:.2f} seconds")
    return


def main():
    pipeline_type = "holography"
    pipeline_start = time.time()
    msger = MessageBoard()
    print()
    msger.welcome_message(pipeline_type)

    main_param_dict, stages = param_init(pipeline_type, msger)

    astrohack_stages = stages[1:6]
    main_param_dict["processing_stage"] = main_param_dict["starting_stage"]

    if main_param_dict["processing_stage"] == stages[0]:
        run_casa_calibration(main_param_dict, msger)
        main_param_dict["processing_stage"] = astrohack_stages[0]

    client = client_initialization(main_param_dict, astrohack_stages)

    if main_param_dict["processing_stage"] in astrohack_stages[:-1]:
        run_astrohack_reduction(main_param_dict, msger)

    if main_param_dict["processing_stage"] == astrohack_stages[-1]:
        run_astrohack_exports(main_param_dict, msger)
        main_param_dict["processing_stage"] = stages[-1]

    if main_param_dict["processing_stage"] == stages[-1]:
        prepare_html_report(main_param_dict, msger)

    if client is not None:
        client.shutdown()

    pipeline_end = time.time()
    msger.goodbye_message(pipeline_type, main_param_dict, pipeline_end - pipeline_start)
