import os
import shutil

from astrohack._utils._panel_classes.base_panel import panel_models
from astrohack._utils._dio import _aips_holog_to_xds, _check_if_file_will_be_overwritten, _check_if_file_exists, _write_meta_data
from astrohack._utils._panel import _panel_chunk
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._param_utils._check_parms import _check_parms, _parm_check_passed
from astrohack._utils._tools import _remove_suffix
from astrohack._utils._dask_graph_tools import _dask_general_compute

from astrohack.mds import AstrohackPanelFile, AstrohackImageFile


def panel(image_name, panel_name=None, clip_type='sigma', clip_level=3, panel_model=None, panel_margins=0.05, ant=None,
          ddi=None, parallel=False, overwrite=False):
    """Analyze holography images to derive panel adjustments

    :param image_name: Input holography data file name. Accepted data formats are the output from ``astrohack.holog.holog`` and AIPS holography data prepackaged using ``astrohack.panel.aips_holog_to_astrohack``.
    :type image_name: str
    :param panel_name: Name of output file; File name will be appended with suffix *.panel.zarr*. Defaults to *basename* of input file plus holography panel file suffix.
    :type panel_name: str, optional
    :param clip_type: Choose the amplitude clipping algorithm: absolute, relative or sigma, default is sigma
    :type clip_type: str, optional
    :param clip_level: Choose level of clipping, default is 3 (appropriate for sigma clipping)
    :type clip_level: float, optional
    :param panel_model: Model of surface fitting function used to fit panel surfaces, None will default to "rigid". Possible models are listed below.
    :type panel_model: str, optional
    :param panel_margins: Relative margin from the edge of the panel used to decide which points are margin points or internal points of each panel. Defaults to 0.05.
    :type panel_margins: float, optional
    :param ant: List of antennas/antenna to be processed, defaults to "all" when None, ex. ea25
    :type ant: list or str, optional
    :param ddi: List of ddis/ddi to be processed, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional
    :param parallel: Run in parallel. Defaults to False.
    :type parallel: bool, optional
    :param overwrite: Overwrite files on disk. Defaults to False.
    :type overwrite: bool, optional

    :return: Holography panel object.
    :rtype: AstrohackPanelFile

    .. _Description:

    **Additional Information**
        Each holography in the input holog image file is processed in the following steps:
        
        .. rubric:: Code Outline
        - Phase image is converted to a physical surface deviation image.
        - A mask of valid signals is created by using the relative cutoff on the amplitude image.
        - From the telescope panel and layout information, an image describing the panel assignment of each pixel is created.
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
        * Corotated Paraboloids: (the two bending axes of the paraboloid are parallel and perpendicular to a radius of the antenna crossing the middle point of the panel):
            - *corotated_scipy*: Paraboloid is fitted using scipy.optimize, robust but slow.
            - *corotated_lst_sq*: Paraboloid is fitted using the linear algebra least squares method, fast but unreliable.
            - *corotated_robust*: Tries corotated_lst_sq, if it diverges falls back to corotated_scipy, fast and robust.
        * Experimental fitting models:
            - *xy_paraboloid*: fitted using scipy.optimize, bending axes are parallel to the x and y axes.
            - *rotated_paraboloid*: fitted using scipy.optimize, bending axes can be rotated by any arbitrary angle.
            - *full_paraboloid_lst_sq*: Full 9 parameter paraboloid fitted using least_squares method, tends to heavily overfit surface irregularities.

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
    Panel object allows the user to access panel data via compound dictionary keys with values, in order of depth, `ant` -> `ddi`. The panel object also provides a `summary()` helper function to list available keys for each file. An outline of the panel object structure is show below:

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

    """

    panel_params = locals()
    panel_params['ant'] = ant
    logger = _get_astrohack_logger()
    fname = 'panel'
    panel_params = _check_panel_parms(fname, panel_params)
    input_params = panel_params.copy()
    # Doubled this entry for compatibility with the factorized antenna ddi loop
    panel_params['filename'] = panel_params['image_name']
    _check_if_file_exists(panel_params['image_name'])
    image_mds = AstrohackImageFile(panel_params['image_name'])
    image_mds._open()
    _check_if_file_will_be_overwritten(panel_params['panel_name'], panel_params['overwrite'])

    if os.path.exists(panel_params['image_name']+'/.aips'):
        panel_params['origin'] = 'AIPS'
        _panel_chunk(panel_params)

    else:
        panel_params['origin'] = 'astrohack'
        if _dask_general_compute(fname, image_mds, _panel_chunk, panel_params, ['ant', 'ddi'], parallel=parallel):
            logger.info(f"[{fname}]: Finished processing")
            output_attr_file = "{name}/{ext}".format(name=panel_params['panel_name'], ext=".panel_input")
            _write_meta_data(output_attr_file, input_params)
            panel_mds = AstrohackPanelFile(panel_params['panel_name'])
            panel_mds._open()
            return panel_mds
        else:
            logger.warning(f"[{fname}]: No data to process")
            return None


def _aips_holog_to_astrohack(amp_image, dev_image, telescope_name, holog_name, overwrite=False):
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
    _check_if_file_exists(amp_image)
    _check_if_file_exists(dev_image)
    _check_if_file_will_be_overwritten(holog_name, overwrite)

    xds = _aips_holog_to_xds(amp_image, dev_image)
    xds.attrs['telescope_name'] = telescope_name
    if os.path.exists(holog_name):
        shutil.rmtree(holog_name, ignore_errors=False, onerror=None)
    xds.to_zarr(holog_name, mode='w', compute=True, consolidated=True)
    aips_mark = open(holog_name+'/.aips', 'w')
    aips_mark.close()


def _check_panel_parms(fname, panel_params):
    """
    Tests inputs to panel function
    Args:
        fname: Caller's name
        panel_params: panel parameters
    """
                          
    #### Parameter Checking ####

    parms_passed = _check_parms(fname, panel_params, 'image_name', [str], default=None)
    base_name = _remove_suffix(panel_params['image_name'], '.image.zarr')
    base_name = _remove_suffix(base_name, '.combine.zarr')
    parms_passed = parms_passed and _check_parms(fname, panel_params, 'panel_name', [str],
                                                 default=base_name+'.panel.zarr')
    parms_passed = parms_passed and _check_parms(fname, panel_params, 'ant', [list, str],
                                                 list_acceptable_data_types=[str], default='all')
    parms_passed = parms_passed and _check_parms(fname, panel_params, 'ddi', [list, int],
                                                 list_acceptable_data_types=[int], default='all')
    parms_passed = parms_passed and _check_parms(fname, panel_params, 'clip_type', [str],
                                                 acceptable_data=['absolute', 'relative', 'sigma'], default='sigma')
    parms_passed = parms_passed and _check_parms(fname, panel_params, 'clip_level', [int, float],
                                                 default=3)
    parms_passed = parms_passed and _check_parms(fname, panel_params, 'panel_model', [str],
                                                 acceptable_data=panel_models, default="rigid")
    parms_passed = parms_passed and _check_parms(fname, panel_params, 'panel_margins', [float],
                                                 acceptable_range=[0, 0.5], default=0.05)
    parms_passed = parms_passed and _check_parms(fname, panel_params, 'parallel', [bool], default=False)
    parms_passed = parms_passed and _check_parms(fname, panel_params, 'overwrite', [bool], default=False)

    _parm_check_passed(fname, parms_passed)

    return panel_params
