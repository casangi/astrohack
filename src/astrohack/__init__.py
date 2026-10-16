import os
from importlib.metadata import version

__version__ = version("astrohack")

# This installs a slick, informational tracebacks logger
from rich.traceback import install
from toolviper.utils.logger import setup_logger

from astrohack.io.beamcut_mds import AstrohackBeamcutFile as AstrohackBeamcutFile
from astrohack.io.dio import open_beamcut as open_beamcut
from astrohack.io.dio import open_holog as open_holog
from astrohack.io.dio import open_image as open_image
from astrohack.io.dio import open_locit as open_locit
from astrohack.io.dio import open_panel as open_panel
from astrohack.io.dio import open_pointing as open_pointing
from astrohack.io.dio import open_position as open_position
from astrohack.io.holog_mds import AstrohackHologFile as AstrohackHologFile
from astrohack.io.image_mds import AstrohackImageFile as AstrohackImageFile
from astrohack.io.locit_mds import AstrohackLocitFile as AstrohackLocitFile
from astrohack.io.panel_mds import AstrohackPanelFile as AstrohackPanelFile
from astrohack.io.point_mds import AstrohackPointFile as AstrohackPointFile
from astrohack.io.position_mds import AstrohackPositionFile as AstrohackPositionFile

from .antenna.telescope import get_proper_telescope as get_proper_telescope
from .beamcut import beamcut as beamcut
from .cassegrain_ray_tracing import (
    apply_holog_phase_fitting_to_rt_xds as apply_holog_phase_fitting_to_rt_xds,
)
from .cassegrain_ray_tracing import (
    cassegrain_ray_tracing_pipeline as cassegrain_ray_tracing_pipeline,
)
from .cassegrain_ray_tracing import (
    create_ray_tracing_telescope_parameter_dict as create_ray_tracing_telescope_parameter_dict,
)
from .cassegrain_ray_tracing import plot_2d_maps_from_rt_xds as plot_2d_maps_from_rt_xds
from .cassegrain_ray_tracing import (
    plot_radial_projection_from_rt_xds as plot_radial_projection_from_rt_xds,
)
from .combine import combine as combine
from .extract_holog import extract_holog as extract_holog
from .extract_holog import generate_holog_obs_dict as generate_holog_obs_dict
from .extract_locit import extract_locit as extract_locit
from .extract_pointing import extract_pointing as extract_pointing
from .fringefit_locit import fringefit_locit as fringefit_locit
from .holog import holog as holog
from .image_comparison_tool import compare_fits_images as compare_fits_images
from .image_comparison_tool import (
    rms_table_from_zarr_datatree as rms_table_from_zarr_datatree,
)
from .locit import locit as locit
from .panel import panel as panel

install(show_locals=False)

if not os.getenv("LOGGER_NAME"):
    os.environ["LOGGER_NAME"] = "astrohack"
    setup_logger(
        logger_name="astrohack",
        log_to_term=True,
        log_to_file=False,
        log_file="astrohack-logfile",
        log_level="INFO",
    )
