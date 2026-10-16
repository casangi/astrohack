import astropy.units as units
import numpy as np
import toolviper.utils.logger as logger
import xarray as xr
from astropy.coordinates import CIRS, SkyCoord
from astropy.time import Time
from casacoretables import tables as ctables

from astrohack.io.locit_mds import AstrohackLocitFile
from astrohack.io.position_mds import AstrohackPositionFile
from astrohack.utils.constants import twopi
from astrohack.utils.conversion import casa_time_to_mjd, convert_unit


def _get_caltable_name(extract_locit_parms):
    if "cal_table" in extract_locit_parms:
        cal_table = extract_locit_parms["cal_table"]
    elif "fringefit_caltable" in extract_locit_parms:
        cal_table = extract_locit_parms["fringefit_caltable"]
    else:
        raise KeyError("No caltable name found in paramter dictionary")
    return cal_table


def extract_antenna_data(
    extract_locit_parms: dict, locit_mds: AstrohackLocitFile | AstrohackPositionFile
):
    """
    Extract antenna information from the ANTENNA sub table of the cal table
    Args:
        extract_locit_parms: input_parameters to extract_locit
        locit_mds: Locit data file object
    """
    cal_table = _get_caltable_name(extract_locit_parms)
    ant_table = ctables.table(
        cal_table + "::ANTENNA",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    ant_off = ant_table.getcol("OFFSET")
    ant_pos = ant_table.getcol("POSITION")
    ant_mnt = ant_table.getcol("MOUNT")
    ant_nam = ant_table.getcol("NAME")
    ant_sta = ant_table.getcol("STATION")
    ant_typ = ant_table.getcol("TYPE")
    ant_table.close()

    n_ant_orig = ant_off.shape[0]
    ant_pos_corrected = ant_pos + ant_off
    ant_rad = np.sqrt(
        ant_pos_corrected[:, 0] ** 2
        + ant_pos_corrected[:, 1] ** 2
        + ant_pos_corrected[:, 2] ** 2
    )
    ant_lat = np.arcsin(ant_pos_corrected[:, 2] / ant_rad)
    ant_lon = -np.arccos(ant_pos_corrected[:, 0] / (ant_rad * np.cos(ant_lat)))

    if extract_locit_parms["ant"] == "all":
        ant_list = ant_nam
    else:
        ant_list = extract_locit_parms["ant"]

    error = False
    for i_ant in range(n_ant_orig):
        this_name = ant_nam[i_ant]
        if this_name in ant_list:
            if ant_mnt[i_ant] != "ALT-AZ":
                logger.error(
                    f"Antenna {this_name} has a non supported mount type: {ant_mnt[i_ant]}"
                )
                error = True
            if ant_typ[i_ant] != "GROUND-BASED":
                error = True
                logger.error(
                    f"Antenna {this_name} is not ground based which is currently not supported"
                )
            if error:
                pass
            else:
                ant_key = f"ant_{this_name}"
                ant_xdtree = xr.DataTree(name=ant_key)
                ant_info = {
                    "id": i_ant,
                    "name": this_name,
                    "station": ant_sta[i_ant],
                    "geocentric_position": ant_pos[i_ant].tolist(),
                    "longitude": float(ant_lon[i_ant]),
                    "latitude": float(ant_lat[i_ant]),
                    "radius": float(ant_rad[i_ant]),
                    "offset": ant_off[i_ant].tolist(),
                }
                ant_xdtree.attrs["antenna_info"] = ant_info
                ant_xdt_name = "/".join([locit_mds.filename, ant_key])
                ant_xdtree.to_zarr(ant_xdt_name, mode="w")

    locit_mds.root.attrs["full_antenna_list"] = ant_nam
    if error:
        msg = "Unsupported antenna characteristics"
        logger.error(msg)
        raise RuntimeError(msg)
    return


def extract_spectral_info(extract_locit_parms):
    """
    Extract spectral information from the SPECTRAL_WINDOW sub table of the cal table
    Args:
        extract_locit_parms: input_parameters to extract_locit

    Returns:
    DDI dictionary
    """

    cal_table = _get_caltable_name(extract_locit_parms)
    spw_table = ctables.table(
        cal_table + "::SPECTRAL_WINDOW",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    ref_freq = spw_table.getcol("REF_FREQUENCY")
    n_chan = spw_table.getcol("NUM_CHAN")
    bandwidth = spw_table.getcol("CHAN_WIDTH")
    spw_names = spw_table.getcol("NAME")
    spw_table.close()
    n_ddi = len(ref_freq)
    error = False
    if extract_locit_parms["ddi"] == "all":
        ddi_list = range(n_ddi)
    else:
        if isinstance(extract_locit_parms["ddi"], int):
            ddi_list = [extract_locit_parms["ddi"]]
        else:
            ddi_list = extract_locit_parms["ddi"]

    ddi_dict = {}
    for i_ddi in ddi_list:
        if i_ddi >= len(n_chan):
            logger.warning(f"DDI {i_ddi} not present in the cal table")
            continue
        if n_chan[i_ddi] != 1:
            error = True
            msg = f"DDI {i_ddi} has {n_chan[i_ddi]}, which is not supported"
            logger.error(msg)
        else:
            ddi_dict[i_ddi] = {
                "id": i_ddi,
                "frequency": ref_freq[i_ddi],
                "bandwidth": bandwidth[i_ddi],
                "name": spw_names[i_ddi],
            }

    if error:
        msg = "Unsupported DDI characteristics"
        logger.error(msg)
        raise RuntimeError(msg)
    return ddi_dict


def extract_source_and_telescope(
    extract_locit_parms, locit_mds: AstrohackLocitFile | AstrohackPositionFile
):
    """
    Extract source and telescope  information from the FIELD and OBSERVATION sub tables of the cal table
    Args:
        extract_locit_parms: input_parameters to extract_locit
        locit_mds: Locit data file object
    """
    cal_table = _get_caltable_name(extract_locit_parms)
    src_table = ctables.table(
        cal_table + "::FIELD",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    src_ids = src_table.getcol("SOURCE_ID")
    phase_center_fk5 = src_table.getcol("PHASE_DIR")[:, 0, :]
    src_name = src_table.getcol("NAME")
    src_table.close()
    n_src = len(src_ids)

    phase_center_fk5[:, 0] = np.where(
        phase_center_fk5[:, 0] < 0,
        phase_center_fk5[:, 0] + twopi,
        phase_center_fk5[:, 0],
    )

    obs_table = ctables.table(
        cal_table + "::OBSERVATION",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    time_range = casa_time_to_mjd(obs_table.getcol("TIME_RANGE")[0])
    telescope_name = obs_table.getcol("TELESCOPE_NAME")[0]
    obs_table.close()

    mid_time = Time((time_range[-1] + time_range[0]) / 2, scale="utc", format="mjd")

    astropy_fk5 = SkyCoord(
        phase_center_fk5[:, 0], phase_center_fk5[:, 1], unit=units.rad, frame="fk5"
    )
    astropy_precessed = astropy_fk5.transform_to(CIRS(obstime=mid_time))
    phase_center_precessed = np.ndarray((n_src, 2))
    phase_center_precessed[:, 0] = astropy_precessed.ra
    phase_center_precessed[:, 1] = astropy_precessed.dec
    phase_center_precessed *= convert_unit("deg", "rad", "trigonometric")

    src_dict = {}
    for i_src in range(n_src):
        src_id = int(src_ids[i_src])
        src_dict[src_id] = {
            "id": src_id,
            "name": src_name[i_src],
            "fk5": phase_center_fk5[i_src].tolist(),
            "precessed": phase_center_precessed[i_src].tolist(),
        }

    locit_mds.root.attrs["source_dict"] = src_dict
    locit_mds.root.attrs["time_range"] = time_range.tolist()
    locit_mds.root.attrs["telescope_name"] = telescope_name


def extract_antenna_phase_gains(extract_locit_parms, ddi_dict, locit_mds):
    """
    Extract antenna based phase gains from the cal table
    Args:
        extract_locit_parms: input_parameters to extract_locit
        ddi_dict: Dictionary with DDI meta information
        locit_mds: Locit data file object
    """

    cal_table = _get_caltable_name(extract_locit_parms)

    telescope_name = locit_mds.root.attrs["telescope_name"]

    main_table = ctables.table(
        cal_table, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )

    antenna1 = main_table.getcol("ANTENNA1")
    antenna2 = main_table.getcol("ANTENNA2")
    gain_time = casa_time_to_mjd(main_table.getcol("TIME"))
    gains = main_table.getcol("CPARAM")
    fields = main_table.getcol("FIELD_ID")
    spw_id = main_table.getcol("SPECTRAL_WINDOW_ID")
    flagged = main_table.getcol("FLAG")
    scans = main_table.getcol("SCAN_NUMBER")
    main_table.close()

    n_gains = len(gains)

    # Ref ant determination and data exclusion based on best refant
    ref_antennas, counts = np.unique(antenna2, return_counts=True)
    n_refant = len(ref_antennas)
    if n_refant > 1:
        i_best_ant = np.argmax(counts)
        fraction_best = counts[i_best_ant] / n_gains
        if fraction_best < 0.5:
            logger.warning(
                f"The best reference Antenna only covers {100 * fraction_best:.1f}% of the data"
            )
        for i_refant in range(n_refant):
            if i_refant != i_best_ant:
                logger.info(
                    f"Discarding gains derived with antenna "
                    f"{locit_mds.root.attrs['full_antenna_list'][ref_antennas[i_refant]]}"
                    f" as reference ({100 * counts[i_refant] / n_gains:.2f}% of the data)"
                )
                sel_refant = antenna2 != ref_antennas[i_refant]
                antenna2 = antenna2[sel_refant]
                antenna1 = antenna1[sel_refant]
                gain_time = gain_time[sel_refant]
                gains = gains[sel_refant]
                fields = fields[sel_refant]
                spw_id = spw_id[sel_refant]
                flagged = flagged[sel_refant]
                scans = scans[sel_refant]
        ref_antenna = ref_antennas[i_best_ant]
    else:
        # No data to discard we can go on and compute the phase gains
        ref_antenna = ref_antennas[0]

    # Calibration tables do not retain a polarization sub table, hence we are going to infer the polarizations present
    # from the telescope
    if "VLA" in telescope_name or telescope_name == "VLBA":
        polarization_scheme = ["R", "L"]
    elif "ALMA" == telescope_name:
        polarization_scheme = ["X", "Y"]
    else:
        msg = f"Unrecognized telescope {extract_locit_parms['telescope_name']}"
        logger.error(msg)
        raise ValueError(msg)

    n_pol = gains.shape[2]
    assert n_pol == 2, logger.error(
        f"Calibration table has {n_pol} polarizations, which is not supported"
    )

    locit_mds.root.attrs["reference_antenna"] = locit_mds.root.attrs[
        "full_antenna_list"
    ][ref_antenna]
    used_sources = []
    phase_gains = np.angle(gains)
    for ant_key, ant_xdtree in locit_mds.items():
        ant_id = ant_xdtree.attrs["antenna_info"]["id"]
        ant_sel = antenna1 == ant_id
        ant_time = gain_time[ant_sel]
        ant_field = fields[ant_sel]
        ant_phase_gains = phase_gains[ant_sel]
        ant_spw_id = spw_id[ant_sel]
        ant_flagged = flagged[ant_sel]
        ant_scans = scans[ant_sel]
        if ant_id == ref_antenna:
            ant_xdtree.attrs["antenna_info"]["reference"] = True
        else:
            ant_xdtree.attrs["antenna_info"]["reference"] = False

        for ddi_id, ddi in ddi_dict.items():
            this_ddi_xds = xr.Dataset()
            ddi_sel = ant_spw_id == ddi_id
            ddi_gains = ant_phase_gains[ddi_sel]
            ddi_time = ant_time[ddi_sel]
            ddi_field = ant_field[ddi_sel]
            ddi_scans = ant_scans[ddi_sel]
            ddi_not_flagged = np.invert(ant_flagged[ddi_sel])

            coords = {}
            for i_pol in range(n_pol):
                time_key = f"p{i_pol}_time"
                coords[time_key] = ddi_time[ddi_not_flagged[:, 0, i_pol]]
                this_ddi_xds[f"P{i_pol}_PHASE_GAINS"] = xr.DataArray(
                    ddi_gains[ddi_not_flagged[:, 0, i_pol], 0, i_pol], dims=time_key
                )
                this_ddi_xds[f"P{i_pol}_FIELD_ID"] = xr.DataArray(
                    ddi_field[ddi_not_flagged[:, 0, i_pol]], dims=time_key
                )
                used_sources.extend(ddi_field[ddi_not_flagged[:, 0, i_pol]])
                this_ddi_xds[f"P{i_pol}_SCANS"] = xr.DataArray(
                    ddi_scans[ddi_not_flagged[:, 0, i_pol]], dims=time_key
                )

            this_ddi_xds.attrs["frequency"] = float(ddi["frequency"])
            this_ddi_xds.attrs["bandwidth"] = ddi["bandwidth"].tolist()
            this_ddi_xds.attrs["polarization_scheme"] = polarization_scheme

            ddi_key = f"ddi_{ddi_id}"
            ddi_xdtree = xr.DataTree(
                dataset=this_ddi_xds.assign_coords(coords), name=ddi_key
            )
            ant_xdtree[ddi_key] = ddi_xdtree

    used_sources = np.unique(np.array(used_sources)).tolist()
    locit_mds.root.attrs["used_sources"] = used_sources
    return
