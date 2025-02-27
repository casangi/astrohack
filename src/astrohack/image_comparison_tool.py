from typing import Union, List
import xarray as xr
import pathlib

import toolviper.utils.logger as logger
import toolviper

from astrohack.core.image_comparison_tool import image_comparison_chunk
from astrohack.utils.graph import compute_graph_from_lists
from astrohack.utils.validation import custom_plots_checker


@toolviper.utils.parameter.validate(
    custom_checker=custom_plots_checker
)
def compare_fits_images(
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
    """
    Compares a set of images to a set of reference images.

    :param image: FITS image or list of FITS images to be compared.
    :type image: list or str

    :param reference_image: FITS image or list of FITS images that serve as references.
    :type reference_image: list or str

    :param telescope_name: Name of the telescope used. Used for masking.
    :type telescope_name: str

    :param destination: Name of directory onto which save plots
    :type destination: str

    :param comparison: Type of comparison to be made between images, "direct" or "scaled", default is "direct".
    :type comparison: str, optional

    :param zarr_container_name: Name of the Zarr container to contain the created datatree, default is None, i.e. \
    DataTree is not saved to disk.
    :type zarr_container_name: str, optional

    :param plot_data: Plot the data array used in the comparison, default is False.
    :type plot_data: bool, optional

    :param plot_percentuals: Plot the residuals in percent of reference image as well, default is False.
    :type plot_percentuals: bool, optional

    :param plot_divided_image: Plot the divided image between Image and its reference, default is False.
    :type plot_divided_image: bool, optional

    :param plot_scatter: Make a scatter plot of the Image against its reference image, default is True.
    :type plot_scatter: bool, optional

    :param export_to_fits: Export created images to FITS files inside destination, default is False.
    :type export_to_fits: bool, optional

    :param colormap: Colormap to be used on image plots, default is "viridis".
    :type colormap: str, optional

    :param dpi: dots per inch to be used in plots, default is 300.
    :type dpi: int, optional

    :param display: Display plots inline or suppress, defaults to True
    :type display: bool, optional

    :param parallel: If True will use an existing astrohack client to do comparison in parallel, default is False
    :type parallel: bool, optional

    :return: DataTree object containing all the comparisons executed
    :rtype: xr.DataTree

    .. _Description:
    Compares pairs of FITS images pixel by pixel using a mask based on telescope parameters to exclude problematic \
    regions such as shadows caused by the secondary mirror or the arms supporting it. By default, 2 products are \
    produced, a plot of the residuals image, i.e. (Reference - Image) and a scatter plot of the Reference against the \
    Image. If necessary a resample of Image is conducted to allow for pixel by pixel comparison.

    .. rubric:: Comparison:
    Two types of comparison between the images are available:
        - *direct*: Where the residuals are simply computed as Reference - Image.
        - *scaled*: Where the residuals are Reference - Factor * Image, with Factor = median(Reference/Image).

    .. rubric:: Plots:
    A plot of the residuals of the comparison is always produced.
    However, a few extra plots can be produced and their production is controlled by the *plot_* parameters, these are:
        - *plot_data*: Activates plotting of the data used in the comparison, default is False as this is the data on \
                       the FITS file.
        - *plot_percentuals*: Activates the plotting of the residuals as a perdentage of the Reference Image, default \
                              is False as this is just another view on the residuals.
        - *plot_divided_image*: Activates the plotting of Reference/Image, default is False. This plot is only \
                                available when using "scaled" comparison.
        - *plot_scatter*: Activates the creation of a scatter plot of Reference vs Image, with a linear regression, \
                          default is True.

    .. rubric:: Storage on disk:
    By default, this function only produces plots, but this can be changed using two parameters:
        - *zarr_container_name*: If this parameter is not None a Zarr container will be created on disk with the \
                                 contents of the produced DataTree.
        - *export_to_fits*: If set to True will produce FITS files of the produced images and store them at \
                            *destination*.

    .. rubric:: Return type:
    This funtion returns a Xarray DataTree containing the Xarray DataSets that represent Image and Reference. The nodes \
    in this DataTree are labelled according to the filenames given as input for easier navigation.
    """

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

    root = xr.DataTree(name='Root')
    for item in result_list:
        tree_node = item[0]
        root = root.assign({tree_node.name: tree_node})

    if zarr_container_name is not None:
        root.to_zarr(zarr_container_name, mode='w', consolidated=True)

    return root
