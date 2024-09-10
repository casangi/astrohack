import os
import pathlib
import shutil
import toolviper.utils.logger as logger
import toolviper.utils.parameter

from astrohack.antenna.panel_fitting import PANEL_MODEL_DICT
from astrohack.utils.fits import aips_holog_to_xds
from astrohack.utils.file import overwrite_file

from astrohack.utils.data import write_meta_data
from astrohack.core.panel import process_panel_chunk
from astrohack.utils.validation import custom_panel_checker
from astrohack.utils.text import get_default_file_name
from astrohack.utils.graph import compute_graph

from astrohack.mds import AstrohackPanelFile, AstrohackImageFile

from typing import Union, List


@toolviper.utils.parameter.validate(    
    custom_checker=custom_panel_checker
)
def panel(
        image_name: str,
        panel_name: str = None,
        clip_type: str = 'sigma',
        clip_level: float = 3.0,
        panel_model: str = "rigid",
        panel_margins: float = 0.05,
        polarization_state: str = 'I',
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[str]] = "all",
        parallel: bool = False,
        overwrite: bool = False
):
    """Analyze holography images to derive panel adjustments

    :param image_name: Input holography data file name. Accepted data formats are the output from \
    ``astrohack.holog.holog`` and AIPS holography data prepackaged using ``astrohack.panel.aips_holog_to_astrohack``.
    :type image_name: str

    :param panel_name: Name of output file; File name will be appended with suffix *.panel.zarr*. Defaults to \
    *basename* of input file plus holography panel file suffix.
    :type panel_name: str, optional

    :param clip_type: Choose the amplitude clipping algorithm: absolute, relative or sigma, default is sigma
    :type clip_type: str, optional

    :param clip_level: Choose level of clipping, default is 3 (appropriate for sigma clipping)
    :type clip_level: float, optional

    :param panel_model: Model of surface fitting function used to fit panel surfaces, None will default to "rigid". \
    Possible models are listed below.
    :type panel_model: str, optional

    :param panel_margins: Relative margin from the edge of the panel used to decide which points are margin points or \
    internal points of each panel. Defaults to 0.05.
    :type panel_margins: float, optional

    :param polarization_state: Select the polarization state over which to run panel, only parallel hands or stokes I \
    should be used, default is I.
    :type polarization_state: str, optional

    :param ant: List of antennas/antenna to be processed, defaults to "all" when None, ex. ea25
    :type ant: list or str, optional

    :param ddi: List of ddi to be processed, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional

    :param parallel: Run in parallel. Defaults to False.
    :type parallel: bool, optional

    :param overwrite: Overwrite files on disk. Defaults to False.
    :type overwrite: bool, optional

    :return: Holography panel object.
    :rtype: AstrohackPanelFile

    .. _Description:

        Each Stokes I aperture image in the input image file is processed in the following steps:
        
        .. rubric:: Code Outline
        - Phase image is converted to a physical surface deviation image.
        - A mask of valid signals is created by using the relative cutoff on the amplitude image.
        - From telescope panel and layout information, an image describing the panel assignment of each pixel is created
        - Using panel image and mask, a list of pixels in each panel is created.
        - Pixels in each panel are divided into two groups: margin pixels and internal pixels.
        - For each panel:
            * Internal pixels are fitted to a surface model.
            * The fitted surface model is used to derive corrections for all pixels in the panel, internal and margins.
            * The fitted surface model is used to derive corrections for the positions of the screws.
        - A corrected deviation image is produced.
        - RMS is computed for both the corrected and uncorrected deviation images.
        - All images produced are stored in the output *.panel.zarr file*.

        .. rubric:: Available panel surface models:
        * AIPS fitting models:
            - *mean*: The panel is corrected by the mean of its samples.
            - *rigid*: The panel samples are fitted to a rigid surface (DEFAULT model).
        * Corotated Paraboloids: (the two bending axes of the paraboloid are parallel and perpendicular to a radius \
        of the antenna crossing the middle point of the panel):
            - *corotated_scipy*: Paraboloid is fitted using scipy.optimize, robust but slow.
            - *corotated_lst_sq*: Paraboloid is fitted using the linear algebra least squares method, fast but \
            unreliable.
            - *corotated_robust*: Tries corotated_lst_sq, if it diverges falls back to corotated_scipy, fast and robust.
        * Experimental fitting models:
            - *xy_paraboloid*: fitted using scipy.optimize, bending axes are parallel to the x and y axes.
            - *rotated_paraboloid*: fitted using scipy.optimize, bending axes can be rotated by any arbitrary angle.
            - *full_paraboloid_lst_sq*: Full 9 parameter paraboloid fitted using least_squares method, tends to \
            heavily overfit surface irregularities.

        .. rubric:: Amplitude clipping:

        In order to produce results of good quality parts of the aperture with low signal (e.g. the shadow of the
        secondary mirror support) a mask is defined based on the amplitude of the aperture. There are 3 methods
        (clip_type parameter) available to define at which level (clip_level) the amplitude is clipped:

        * absolute: In this method the clipping value is taken directly from the clip_level parameter, e.g.:
                    if the user calls `panel(..., clip_type='absolute', clip_level=3.5)` everything below 3.5 in
                    amplitude will be clipped
        * relative: In this method the clipping value is derived from the amplitude maximum, e.g.: if the user calls
                    `panel(..., clip_type='relative', clip_level=0.2) everything below 20% of the maximum amplitude will
                    be clipped
        * sigma;    In this method the clipping value is computed from the RMS noise in the amplitude outside the
                    physical dish, e.g.: if the user calls `panel(clip_type='sigma', clip_level=3)` everything below 3
                    times the RMS noise in amplitude will be clipped.

        The default clipping is set to 3 sigma.


    .. _Description:
    **AstrohackPanelFile**
    Panel object allows the user to access panel data via compound dictionary keys with values, in order of depth, \
    `ant` -> `ddi`. The panel object also provides a `summary()` helper function to list available keys for each file.\
     An outline of the panel object structure is show below:

    .. parsed-literal::
        panel_mds = 
        {
            ant_0:{
                ddi_0: panel_ds,
                 ⋮               
                ddi_m: panel_ds
            },
            ⋮
            ant_n: …
        }

    **Example Usage**

    .. parsed-literal::
        from astrohack.panel import panel

        # Fit the panels in the aperture image by using a rigid panel model and
        # excluding the border 5% of each panel from the fitting.
        panel_mds = panel(
            "myholo.image.zarr",
            panel_model='rigid',
            panel_margin=0.05
        )

        # fit the panels in the aperture image by using a rigid panel model and
        # excluding points in the aperture image which have an amplitude that is less than 20% of the peak amplitude.
        panel_mds = panel(
            "myholo.image.zarr",
            clip_type='relative',
            clip_level=0.2
        )

    """
    # Doing this here allows it to get captured by locals()
    if panel_name is None:
        panel_name = get_default_file_name(input_file=image_name, output_type='.panel.zarr')

    panel_params = locals()

    input_params = panel_params.copy()
    assert pathlib.Path(panel_params['image_name']).exists() is True, (
        logger.error(f"File {panel_params['image_name']} does not exists.")
    )

    image_mds = AstrohackImageFile(panel_params['image_name'])
    image_mds.open()

    overwrite_file(panel_params['panel_name'], panel_params['overwrite'])

    if PANEL_MODEL_DICT[panel_model]['experimental']:
        logger.warning(f'Using experimental panel fitting model {panel_model}')

    if os.path.exists(panel_params['image_name'] + '/.aips'):
        panel_params['origin'] = 'AIPS'
        process_panel_chunk(panel_params)

    else:
        panel_params['origin'] = 'astrohack'
        if compute_graph(image_mds, process_panel_chunk, panel_params, ['ant', 'ddi'], parallel=parallel):
            logger.info("Finished processing")
            output_attr_file = "{name}/{ext}".format(name=panel_params['panel_name'], ext=".panel_input")
            write_meta_data(output_attr_file, input_params)
            panel_mds = AstrohackPanelFile(panel_params['panel_name'])
            panel_mds.open()

            return panel_mds
        else:
            logger.warning("No data to process")
            return None


def _aips_holog_to_astrohack(
        amp_image: str,
        dev_image: str,
        telescope_name: str,
        holog_name: str,
        overwrite: bool = False
):
    """
    Package AIPS HOLOG products in a .image.zarr file compatible with astrohack.panel.panel

    This function reads amplitude and deviation FITS files produced by AIPS's HOLOG task and transfers their data onto a
    .image.zarr file that can be read by panel.
    Most of the metadata can be inferred from the FITS headers, but it remains necessary to specify the telescope name
    to be included on the .image.zarr file

    Args:
        amp_image: Full path to amplitude image
        dev_image: Full path to deviation image
        telescope_name: Telescope name to be added to the .zarr file
        holog_name: Name of the output .zarr file
        overwrite: Overwrite previous file of same name?
    """
    assert pathlib.Path(amp_image).exists() is True, logger.error(f'File {amp_image} does not exists.')
    assert pathlib.Path(dev_image).exists() is True, logger.error(f'File {dev_image} does not exists.')

    overwrite_file(holog_name, overwrite)

    xds = aips_holog_to_xds(amp_image, dev_image)
    xds.attrs['telescope_name'] = telescope_name

    if pathlib.Path(holog_name).exists():
        shutil.rmtree(holog_name, ignore_errors=False, onerror=None)

    xds.to_zarr(holog_name, mode='w', compute=True, consolidated=True)
    aips_mark = open(holog_name + '/.aips', 'w')
    aips_mark.close()
