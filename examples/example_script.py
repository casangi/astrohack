import os
import distributed

import numpy as np
import matplotlib.pyplot as plt

from toolviper.dask.client.client import local_client

from astrohack.extract_pointing import extract_pointing
from astrohack.extract_holog import generate_holog_obs_dict
from astrohack.extract_holog import extract_holog
from astrohack.holog import holog

from astrohack.utils._logger._astrohack_logger import _get_astrohack_logger


def process():
    # < -------------------------------------------------------------------- >
    #    Here is where you modify your file path for all local variables.
    # < -------------------------------------------------------------------- >
    ms_file = "/lustre/cv/users/jhoskins/otf33.ms"
    point_name = "data/otf.point.zarr"
    holog_name = "data/otf.holog.zarr"
    image_name = "data/otf.image.zarr"

    # Extract pointing dictionary
    point_mds = extract_pointing(
        ms_name=ms_file, point_name=point_name, parallel=True, overwrite=True
    )

    # Generate holog_obs_dict
    holog_obs_dict = generate_holog_obs_dict(ms_name=ms_file, point_name=point_name)

    # Trimming example
    trimmed_dict = holog_obs_dict.select(key="ddi", value=1, inplace=False)

    # Choose baseline
    trimmed_dict = trimmed_dict.select(
        key="baseline", value="ea15", n_baselines=3, inplace=False
    )

    trimmed_dict.print()

    # Extract holography
    holog_mds = extract_holog(
        ms_name=ms_file,
        point_name=point_name,
        holog_name=holog_name,
        holog_obs_dict=trimmed_dict,
        data_column="CORRECTED_DATA",
        parallel=True,
        overwrite=True,
    )

    # Create image
    image_mds = holog(
        holog_name=holog_name,
        image_name=image_name,
        grid_size=np.array([25, 25]),
        overwrite=True,
        parallel=True,
    )

    # Plot image
    image_mds.plot_apertures(destination="plots", ant="ea15", display=True)

    plt.show()


if __name__ == "__main__":
    # Setup the client to run
    log_params = {"log_level": "DEBUG"}

    client = astrohack_local_client(cores=2, memory_limit="24GB", log_params=log_params)

    logger = _get_astrohack_logger()
    logger.info(client.dashboard_link)

    # Process data
    process()

    client.shutdown()
