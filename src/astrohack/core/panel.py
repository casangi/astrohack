import toolviper.utils.logger as logger

from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.antenna.telescope import Telescope
from astrohack.utils import create_dataset_label, data_from_version_needs_patch


def process_panel_chunk(panel_chunk_params):
    """
    Process a chunk of the holographies, usually a chunk consists of an antenna over a ddi
    Args:
        panel_chunk_params: dictionary of inputs
    """

    clip_level = panel_chunk_params["clip_level"]
    ddi = panel_chunk_params["this_ddi"]
    antenna = panel_chunk_params["this_ant"]
    inputxds = panel_chunk_params["xds_data"]
    logger.info(f"processing {create_dataset_label(antenna, ddi)}")
    telescope = Telescope.from_xds(inputxds)
    if isinstance(clip_level, dict):
        ant_name = antenna.split("_")[1]
        ddi_name = int(ddi.split("_")[1])
        try:
            clip_level = clip_level[ant_name][ddi_name]
        except KeyError:
            msg = f"Antenna {ant_name} and DDI {ddi_name} combination not found in clip_level dictionary"
            logger.error(msg)
            raise Exception(msg)

    needs_phase_wrapping_patch = data_from_version_needs_patch(
        panel_chunk_params["version"], "0.5.10"
    )

    surface = AntennaSurface(
        inputxds,
        telescope,
        clip_type=panel_chunk_params["clip_type"],
        pol_state=panel_chunk_params["polarization_state"],
        clip_level=clip_level,
        pmodel=panel_chunk_params["panel_model"],
        panel_margins=panel_chunk_params["panel_margins"],
        patch_phase=needs_phase_wrapping_patch,
        exclude_shadows=panel_chunk_params["exclude_shadows"],
    )

    surface.compile_panel_points()
    surface.fit_surface()
    surface.correct_surface()

    xds_name = panel_chunk_params["panel_name"] + f"/{antenna}/{ddi}"
    xds = surface.export_xds()
    xds.to_zarr(xds_name, mode="w")
