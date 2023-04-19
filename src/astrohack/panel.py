import os
import dask
import xarray as xr
import shutil

from astrohack._classes.antenna_surface import AntennaSurface
from astrohack._classes.telescope import Telescope
from astrohack._classes.base_panel import panel_models
from astrohack._utils._io import _load_image_xds, _aips_holog_to_xds, check_if_file_will_be_overwritten, check_if_file_exists
from astrohack._utils._panel import _external_to_internal_parameters, _correct_phase
import numpy as np

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._utils import _remove_suffix

from astrohack._utils._dio import AstrohackPanelFile


def panel(image_name, panel_name=None, cutoff=0.2, panel_model=None, panel_margins=0.2, parallel=False, sel_ddi=None,
          overwrite=False):
    """Analyze holography images to derive panel adjustments

    :param image_name: Input holography data file name. Accepted data formats are the output from ``astrohack.holog.holog`` and AIPS holography data prepackaged using ``astrohack.panel.aips_holog_to_astrohack``.
    :type image_name: str
    :param panel_name: Name of output file; File name will be appended with suffix *.panel.zarr*. Defaults to *basename* of input file plus holography panel file suffix.
    :type panel_name: str, optional
    :param cutoff: Relative amplitude cut-off which defines fitting mask. Defaults to 0.2.
    :type cutoff: float, optional
    :param panel_model: Model of surface fitting function used to fit panel surfaces, None will default to "rigid". Possible models are listed below.
    :type panel_model: str, optional
    :param panel_margins: Relative margin from the edge of the panel used to decide which points are margin points or internal points of each panel. Defaults to 0.2.
    :type panel_margins: float, optional
    :param parallel: Run in parallel. Defaults to False.
    :type parallel: bool, optional
    :param sel_ddi: List of DDIs to be processed. None will use all DDIs. Defaults to None.
    :type sel_ddi: list, optional
    :param overwrite: Overwrite files on disk. Defaults to False.
    :type overwrite: bool, optional

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


    """
    
    logger = _get_astrohack_logger()
    
    panel_params = _check_panel_parms(image_name, panel_name, cutoff, panel_model, panel_margins, parallel, sel_ddi,
                                      overwrite)
          
    check_if_file_exists(panel_params['image_name'])
    check_if_file_will_be_overwritten(panel_params['panel_name'], panel_params['overwrite'])

    if os.path.exists(panel_params['image_name']+'/.aips'):
        panel_params['origin'] = 'AIPS'
        _panel_chunk(panel_params)

    else:
        panel_chunk_params = panel_params
        panel_chunk_params['origin'] = 'astrohack'
        delayed_list = []

        antennae = os.listdir(panel_chunk_params['image_name'])
        count = 0
        for antenna in antennae:
            if 'ant_' in antenna:
                panel_chunk_params['antenna'] = antenna
                
                if panel_chunk_params['sel_ddi'] == "all":
                    panel_chunk_params['sel_ddi'] = os.listdir(panel_chunk_params['image_name']+'/'+antenna)

                for ddi in panel_chunk_params['sel_ddi'] :
                    if 'ddi_' in ddi:
                        logger.info(f"Processing {ddi} for {antenna}")
                        panel_chunk_params['ddi'] = ddi
                        if parallel:
                            delayed_list.append(dask.delayed(_panel_chunk)(dask.delayed(panel_chunk_params)))
                        else:
                            _panel_chunk(panel_chunk_params)
                        count += 1
        if parallel:
            dask.compute(delayed_list)

        if count == 0:
            logger.warning("No data to process")
        else:
            logger.info("Panel finished processing")
            
            panel_mds = AstrohackPanelFile(panel_chunk_params['panel_name'])
            panel_mds.open()
            return panel_mds


def _panel_chunk(panel_chunk_params):
    """
    Process a chunk of the holographies, usually a chunk consists of an antenna over a ddi
    Args:
        panel_chunk_params: dictionary of inputs
    """
    if panel_chunk_params['origin'] == 'AIPS':
        inputxds = xr.open_zarr(panel_chunk_params['image_name'])
        telescope = Telescope(inputxds.attrs['telescope_name'])
        panel_chunk_params['antenna'] = inputxds.attrs['ant_name']

    else:
        inputxds = _load_image_xds(panel_chunk_params['image_name'],
                                   panel_chunk_params['antenna'],
                                   panel_chunk_params['ddi'],
                                   dask_load=False)

        inputxds.attrs['AIPS'] = False

        if inputxds.attrs['telescope_name'] == "ALMA":
            tname = inputxds.attrs['telescope_name']+'_'+inputxds.attrs['ant_name'][0:2]
            telescope = Telescope(tname)
        elif inputxds.attrs['telescope_name'] == "EVLA":
            tname = "VLA"
            telescope = Telescope(tname)
        else:
            raise ValueError('Unsuported telescope {0:s}'.format(inputxds.attrs['telescope_name']))

    surface = AntennaSurface(inputxds, telescope, panel_chunk_params['cutoff'], panel_chunk_params['panel_kind'],
                             panel_margins=panel_chunk_params['panel_margins'])

    surface.compile_panel_points()
    surface.fit_surface()
    surface.correct_surface()
    
    xds_name = panel_chunk_params['panel_name'] + '/' + panel_chunk_params['antenna'] + '/' + panel_chunk_params['ddi']
    xds = surface.export_xds()
    xds.to_zarr(xds_name, mode='w')


def _create_phase_model(npix, parameters, wavelength, telescope, cellxy):
    """
    Create a phase model with npix by npix size according to the given parameters
    Args:
        npix: Number of pixels in each size of the model
        parameters: Parameters for the phase model in the units described in _phase_fitting
        wavelength: Observing wavelength, in meters
        telescope: Telescope object containing the optics parameters
        cellxy: Map cell spacing, in meters

    Returns:

    """
    iNPARameters = _external_to_internal_parameters(parameters, wavelength, telescope, cellxy)
    dummyphase = np.zeros((npix, npix))

    _, model = _correct_phase(dummyphase, cellxy, iNPARameters, telescope.magnification, telescope.focus,
                              telescope.surp_slope)
    return model


def aips_holog_to_astrohack(amp_image, dev_image, telescope_name, holog_name, overwrite=False):
    """
    Package AIPS HOLOG products in a .image.zarr file compatible with astrohack.panel.panel

    This function reads amplitude and deviation FITS files produced by AIPS's HOLOG task and transfers their data onto a
    .image.zarr file that can be read by panel.
    Most of the meta data can be inferred from the FITS headers, but it remains necessary to specify the telescope name
    to be included on the .image.zarr file

    Args:
        amp_image: Full path to amplitude image
        dev_image: Full path to deviation image
        telescope_name: Telescope name to be added to the .zarr file
        holog_name: Name of the output .zarr file
        overwrite: Overwrite previous file of same name?
    """
    check_if_file_exists(amp_image)
    check_if_file_exists(dev_image)
    check_if_file_will_be_overwritten(holog_name, overwrite)

    xds = _aips_holog_to_xds(amp_image, dev_image)
    xds.attrs['telescope_name'] = telescope_name
    if os.path.exists(holog_name):
        shutil.rmtree(holog_name, ignore_errors=False, onerror=None)
    xds.to_zarr(holog_name, mode='w', compute=True, consolidated=True)
    aips_mark = open(holog_name+'/.aips', 'w')
    aips_mark.close()


def _check_panel_parms(image_name, panel_name, cutoff, panel_kind, panel_margins, parallel, sel_ddi, overwrite):
    """
    Tests inputs to panel function
    Args:
        image_name: Input holography data, can be from astrohack.holog, but also preprocessed AIPS data
        panel_name: Name for the output directory structure containing the products

        cutoff: Cut off in amplitude for the physical deviation fitting, None means 20%
        panel_kind: Type of fitting function used to fit panel surfaces, defaults to corotated_paraboloid for ringed
                    telescopes
        parallel: Run chunks of processing in parallel
        panel_margins: Margin to be ignored at edges of panels when fitting
        sel_ddi: Which DDIs are to be processed by panel, None means all of them
        overwrite: Overwrite previous hack_file of same name?
    """

    panel_params = {'image_name': image_name,
                    'panel_name': panel_name,
                    'cutoff': cutoff,
                    'panel_kind': panel_kind,
                    'panel_margins': panel_margins,
                    'parallel': parallel,
                    'sel_ddi': sel_ddi,
                    'overwrite': overwrite
                    }
                          
    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    parms_passed = True
    
    parms_passed = parms_passed and _check_parms(panel_params, 'image_name', [str], default=None)
    base_name = _remove_suffix(panel_params['image_name'], '.image.zarr')
    parms_passed = parms_passed and _check_parms(panel_params, 'panel_name', [str], default=base_name+'.panel.zarr')
    parms_passed = parms_passed and _check_parms(panel_params, 'cutoff', [float], acceptable_range=[0, 1], default=0.2)
    parms_passed = parms_passed and _check_parms(panel_params, 'panel_kind', [str], acceptable_data=panel_models, default="rigid")
    parms_passed = parms_passed and _check_parms(panel_params, 'panel_margins', [float], acceptable_range=[0, 0.5], default=0.2)
    parms_passed = parms_passed and _check_parms(panel_params, 'parallel', [bool], default=False)
    parms_passed = parms_passed and _check_parms(panel_params, 'sel_ddi', [list, np.array], list_acceptable_data_types=[int, np.int], default='all')
    parms_passed = parms_passed and _check_parms(panel_params, 'overwrite', [bool], default=False)

    if not parms_passed:
        logger.error("panel parameter checking failed.")
        raise Exception("panel parameter checking failed.")
    
    return panel_params


