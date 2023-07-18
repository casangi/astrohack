import os
from astrohack.gdown_utils import download
from astrohack import astrohack_local_client
from astrohack.extract_holog import extract_holog

from astrohack.holog import holog

def demo_setup():
    astrohack_local_client(cores=2, memory_limit="8GB")
    download(
            "J1924-2914.ms.calibrated.split.SPW3", folder=os.getcwd(), unpack=True
    )
    extract_holog(
        ms_name="J1924-2914.ms.calibrated.split.SPW3",
        holog_name="alma.split.holog.zarr",
        data_column="DATA",
        parallel=True,
        overwrite=True,
        reuse_point_zarr=False,
    )

def holog_demo():
    holog(
        holog_name="alma.split.holog.zarr",
        padding_factor=50,
        grid_interpolation_mode="linear",
        chan_average=True,
        scan_average=True,
        overwrite=True,
        phase_fit=True,
        apply_mask=True,
        to_stokes=True,
        parallel=True,
    )
    
def test_holog_with_setup(benchmark):
    benchmark.pedantic(holog_demo, setup=demo_setup, iterations=1, rounds=5)
