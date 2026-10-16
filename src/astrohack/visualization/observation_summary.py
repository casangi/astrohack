from astrohack.utils.graph import create_and_execute_graphs_for_outputs
from astrohack.utils.text import format_observation_summary, lnbr, make_header, spc


def generate_observation_summary(
    mds_object, param_dict: dict, summary_type: str, key_order: list[str]
):
    param_dict["dtype"] = summary_type
    execution, summary_list = create_and_execute_graphs_for_outputs(
        mds_object=mds_object,
        chunk_function=_generate_observation_summary_chunk,
        key_order=key_order,
        param_dict=param_dict,
        fetch_returns=True,
    )
    if execution:
        full_summary = "".join(summary_list)
        with open(param_dict["summary_file"], "w") as output_file:
            output_file.write(full_summary)
        if param_dict["print_summary"]:
            print(full_summary)


def _generate_observation_summary_chunk(parm_dict):
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    data_type = parm_dict["dtype"]
    xds = parm_dict["xdt_data"]
    obs_sum = xds.attrs["summary"]
    tab_size = parm_dict["tab_size"]

    tab_count = 1

    if data_type == "holog":
        map_id = parm_dict["this_map"]
        header = f"{antenna}, {ddi}, {map_id}"
    else:
        header = f"{antenna}, {ddi}"

    outstr = make_header(header, "#", 60, 3)

    outstr += (
        format_observation_summary(
            obs_sum,
            tab_size,
            tab_count,
            az_el_key=parm_dict["az_el_key"],
            phase_center_unit=parm_dict["phase_center_unit"],
            az_el_unit=parm_dict["az_el_unit"],
            time_format=parm_dict["time_format"],
        )
        + lnbr
    )

    if data_type == "beamcut":
        for cut in xds.children.values():
            outstr += f"{tab_count * tab_size * spc}{cut.name}:{lnbr}"
            outstr += f"{(tab_count + 1) * tab_size * spc}{cut.attrs['direction']} at {cut.attrs['time_string']} UTC{2 * lnbr}"

    return outstr
