from typing import Union, List
import xarray as xr
import pathlib

import toolviper.utils.logger as logger

from astrohack.core.image_compare_tool import image_comparison_chunk
from astrohack.utils.graph import compute_graph_from_lists


def compare_fits_image(
        image: Union[str, List[str]],
        reference_image: Union[str, List[str]],
        telescope_name: str,
        destination: str,
        comparison: str = 'direct',
        zarr_container_name: str = None,
        plot_data: bool = False,
        plot_percentuals: bool = False,
        plot_divided_image: bool = False,
        plot_scatter: bool = True,
        export_to_fits: bool = False,
        colormap: str = 'viridis',
        dpi: int = 300,
        display: bool = False,
        parallel: bool = False
):

    if isinstance(image, str):
        image = [image]
    if isinstance(reference_image, str):
        reference_image = [reference_image]
    if len(image) != len(reference_image):
        msg = 'List of reference images has a different size from the list of images'
        logger.error(msg)
        return

    param_dict = locals()
    pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)

    result_list = compute_graph_from_lists(param_dict, image_comparison_chunk, ['image', 'reference_image'], parallel)

    if zarr_container_name is not None:
        root = xr.DataTree(name='Root')
        for item in result_list:
            tree_node = item[0]
            root = root.assign({tree_node.name: tree_node})

        root.to_zarr(zarr_container_name, mode='w', consolidated=True)
