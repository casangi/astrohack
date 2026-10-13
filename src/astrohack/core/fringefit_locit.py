import shutil

import numpy as np
from casacoretables import tables as ctables

from astrohack import AstrohackPositionFile
from astrohack.core.locit import (
    _build_coordinate_array,
    _filter_data_on_elevation_and_scans,
)
import toolviper.utils.logger as logger


def fringefit_locit_looping_dict(
    locit_parms: dict, full_antenna_list: list
) -> tuple[dict, str]:
    """
    Create looping dictionary for fringefit locit
    Args:
        locit_parms: Input parameters to fringefit locit
        full_antenna_list: Full list of antenna names

    Returns:
        tuple containing: the looping dictionary and the reference antenna name
    """
    fringefit_caltable = locit_parms["fringefit_caltable"]
    main_table = ctables.table(
        f"{fringefit_caltable}",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    times = main_table.getcol("TIME")
    ant1 = main_table.getcol("ANTENNA1")
    ant2 = main_table.getcol("ANTENNA2")
    fparam = main_table.getcol("FPARAM")
    spw = main_table.getcol("SPECTRAL_WINDOW_ID")
    field = main_table.getcol("FIELD_ID")
    scans = main_table.getcol("SCAN_NUMBER")

    delays = fparam[:, 0, 1::4] * 1e-9
    if locit_parms["ant"] == "all":
        looping_ant_list = full_antenna_list
    else:
        if isinstance(locit_parms["ant"], str):
            user_ant_list = [locit_parms["ant"]]
        else:
            user_ant_list = locit_parms["ant"]
        looping_ant_list = []
        for ant_name in user_ant_list:
            if ant_name in full_antenna_list:
                looping_ant_list.append(ant_name)

    unq_refants = np.unique(ant2)
    refant_id = unq_refants[0]
    if unq_refants.size != 1:
        logger.warning(
            "More than one refant not supported dropping data with alternative reference antennas"
        )
    ant2_sel = ant2 == refant_id
    refant_name = full_antenna_list[refant_id]
    looping_dict = {}
    unq_ants_in_data = np.unique(ant1)
    for ant_id in unq_ants_in_data:
        ant_name = full_antenna_list[ant_id]
        if ant_name not in looping_ant_list:
            continue
        ant_key = f"ant_{ant_name}"
        ant_selection = np.logical_and(ant1 == ant_id, ant2_sel)

        if np.sum(delays[ant_selection]) > 0:
            this_ant_data = {
                "time": times[ant_selection],
                "delays": delays[ant_selection],  # convert to sec
                "fields": field[ant_selection],
                "spw": spw[ant_selection],
                "scans": scans[ant_selection],
            }
            looping_dict[ant_key] = this_ant_data
        else:
            logger.warning(f"No valid delay data for {ant_name}")
            shutil.rmtree(f"{locit_parms['position_name']}/{ant_key}")
    return looping_dict, refant_name


def _match_delays_to_coordinates(
    locit_parms: dict,
    field_dict: dict,
    ant_info: dict,
    delay_dict: dict,
    init_time: float,
    ddi_dict: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, list]:
    """
    Match delays to coordinates
    Args:
        locit_parms: Input parameters for fringefit locit.
        field_dict: Dictionary containing the sources in the fringefit cal table.
        ant_info: Dictionary containing the information about the antenna being processed.
        delay_dict: Dictionary containing the delay information for the antenna being processed.
        init_time: Initial time of the observations.
        ddi_dict: Dictionary containing the spectral window information.

    Returns:
        a size 6 tuple containing: the [n,4] coordinates array, [n] delay array, [n] LST array, [n] scan array, \
        Elevation limit in rad, list of used spws.
    """
    user_pol_sel = locit_parms["polarization"]

    if user_pol_sel == "both":
        pol_sel = [0, 1]
    elif user_pol_sel == "R":
        pol_sel = [0]
    elif user_pol_sel == "L":
        pol_sel = [1]
    else:
        raise ValueError(f"Polarization selection ({user_pol_sel}) not recognized")

    spw = delay_dict["spw"]
    ddi_sel = np.full_like(spw, False)
    used_ddis = []
    for ddi in ddi_dict.keys():
        this_ddi_sel = spw == ddi
        if np.sum(this_ddi_sel) > 0:
            used_ddis.append(ddi)
        ddi_sel = np.logical_or(this_ddi_sel, ddi_sel)

    ant_time = delay_dict["time"][ddi_sel] / 86400
    ant_fields = delay_dict["fields"][ddi_sel]
    ant_delays = delay_dict["delays"][ddi_sel]
    ant_scans = delay_dict["scans"][ddi_sel]

    n_dumps_coordinates, n_dumps_lst = _build_coordinate_array(
        ant_info, ant_time, ant_fields, field_dict
    )

    n_rows = ant_time.shape[0]
    n_pol = len(pol_sel)
    coordinate_array = np.zeros((4, n_pol * n_rows))
    delay_array = np.zeros(n_pol * n_rows)
    lst_array = np.zeros(n_pol * n_rows)
    scan_array = np.zeros(n_pol * n_rows, dtype=int)
    for i_pol, pol_id in enumerate(pol_sel):
        f_row = i_pol * n_rows
        l_row = (i_pol + 1) * n_rows
        coordinate_array[:, f_row:l_row] = n_dumps_coordinates
        delay_array[f_row:l_row] = ant_delays[:, pol_id]
        lst_array[f_row:l_row] = n_dumps_lst
        scan_array[f_row:l_row] = ant_scans

    elevation_limit_rad, scan_array, delay_array, coordinate_array, lst_array = (
        _filter_data_on_elevation_and_scans(
            locit_parms["elevation_limit"],
            locit_parms["exclude_scans"],
            coordinate_array,
            lst_array,
            scan_array,
            delay_array,
        )
    )

    return (
        coordinate_array,
        delay_array,
        lst_array,
        scan_array,
        elevation_limit_rad,
        used_ddis,
    )


def _get_average_freq(ddi_dict: dict, used_ddis: list) -> float:
    """
    Compute the average frequency among the used ddis weighted by bandwidth.
    Args:
        ddi_dict: Dictionary containing spw information
        used_ddis: List of effectively used ddis

    Returns:
        average_freq: float
    """
    freqs = []
    bws = []
    for key, value in ddi_dict.items():
        if key in used_ddis:
            freqs.append(value["frequency"])
            bws.append(value["bandwidth"][0])
    average_freq = float(np.average(freqs, weights=bws))
    return average_freq


def fringefit_locit_chunk(locit_parms: dict, output_mds: AstrohackPositionFile):
    """
    Chunk function for fringefit locit
    Args:
        locit_parms: input parameters for fringefit locit
        output_mds: The output AstrohackPositionFile onto which to add this antenna's xds.

    Returns:
        Antenna xds saved to disk.
    """
    from astrohack.core.locit import (
        _solve_linear_algebra,
        _solve_scipy_optimize_curve_fit,
        _compute_chi_squared,
        _create_output_xds,
    )

    ddi_dict = locit_parms["ddi_dict"]
    ant_order = [locit_parms["this_ant"]]
    current_xds = output_mds.open_subset(ant_order)
    antenna_info = current_xds.attrs["antenna_info"]
    src_dict = output_mds.root.attrs["source_dict"]
    ant_name = antenna_info["name"]
    logger.info(f"Processing {ant_name}")

    delay_dict = locit_parms["dic_data"]
    init_time = output_mds.root.attrs["time_range"][0]
    coordinates, delays, lst, scans, el_limit, used_ddis = _match_delays_to_coordinates(
        locit_parms, src_dict, antenna_info, delay_dict, init_time, ddi_dict
    )
    if coordinates.size == 0:
        logger.warning(f"Data selection excludes all data for {ant_name}")
        shutil.rmtree(f"{output_mds.filename}/{locit_parms['this_ant']}")
        return

    average_freq = _get_average_freq(ddi_dict, used_ddis)
    fit_kterm = locit_parms["fit_kterm"]
    fit_delay_rate = locit_parms["fit_delay_rate"]
    if locit_parms["fit_engine"] == "linear algebra":
        fit_func = _solve_linear_algebra
    else:
        fit_func = _solve_scipy_optimize_curve_fit

    fit, variance = fit_func(coordinates, delays, fit_kterm, fit_delay_rate)
    model, chi2 = _compute_chi_squared(
        delays, fit, coordinates, fit_kterm, fit_delay_rate
    )

    current_xds = _create_output_xds(
        coordinates,
        lst,
        delays,
        scans,
        fit,
        variance,
        chi2,
        model,
        locit_parms,
        average_freq,
        el_limit,
        antenna_info,
    )
    output_mds.add_node(current_xds, ant_order)
    return
