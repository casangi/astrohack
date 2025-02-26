from typing import Union, List
import xarray as xr
import pathlib

import toolviper.utils.logger as logger

from astrohack.core.image_compare_tool import image_comparison_chunk


def compare_fits_image(
        image: Union[str, List[str]],
        reference_image: Union[str, List[str]],
        telescope_name: str,
        destination: str,
        comparison: str = 'direct',
        plot_data: bool = False,
        plot_percentuals: bool = False,
        plot_divided_image: bool = False,
        plot_scatter: bool = True,
        export_to_fits: bool = False,
        colormap: str = 'viridis',
        dpi: int = 300,
        display: bool = False
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
    for i_img in range(len(image)):
        param_dict['this_image'] = image[i_img]
        param_dict['this_reference_image'] = reference_image[i_img]
        image_comparison_chunk(param_dict)
