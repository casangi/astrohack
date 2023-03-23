import os
import dask
import xarray as xr

from astrohack._classes.antenna_surface import AntennaSurface
from astrohack._classes.telescope import Telescope
from astrohack._utils._io import _load_image_xds
from astrohack._utils._system_message import info, warning
from astrohack._utils._panel import _external_to_internal_parameters, _correct_phase
import numpy as np
import time


def panel(holog_image, outfile, aipsdata=False, telescope=None, cutoff=None, panel_kind=None, basename=None, unit='mm',
          panel_margins=0.2, save_mask=False, save_deviations=False, save_phase=False, parallel=True, ddis=None):
    """
    Process holographies to produce screw adjustments for panels, several data products are also produced in the process
    Args:
        holog_image: Input holography data, can be from astrohack.holog, but also preprocessed AIPS data
        outfile: Name for the output directory structure containing the products
        aipsdata: Is input data from AIPS, if so ony a single antenna can be processed at a time
        telescope: Name of the telescope used, can be derived from the holography dataset
        cutoff: Cut off in amplitude for the physical deviation fitting, None means 20%
        panel_kind: Type of fitting function used to fit panel surfaces, defaults to corotated_paraboloid for ringed
                    telescopes
        basename: Name for subfolders in the directory structure, defaults to telescope name if None
        unit: Unit for panel adjustments
        save_mask: Save plot of the mask derived from amplitude cutoff to a png file
        save_deviations: Save plot of physical deviations to a png file
        save_phase: Save plot of phases to a png file
        parallel: Run chunks of processing in parallel
        panel_margins: Margin to be ignored at edges of panels when fitting
        ddis: Which DDIs are to be processed by panel, None means all of them
    """

    outfile += '.panel.zarr'
    panel_chunk_params = {'holog_image': holog_image,
                          'unit': unit,
                          'panel_kind': panel_kind,
                          'cutoff': cutoff,
                          'save_mask': save_mask,
                          'save_deviations': save_deviations,
                          'save_phase': save_phase,
                          'telescope': telescope,
                          'basename': basename,
                          'outfile': outfile,
                          'panel_margins': panel_margins,
                          }
    os.makedirs(name=outfile,  exist_ok=True)

    if aipsdata:
        if telescope is None:
            raise Exception('For AIPS data a telescope must be specified')
        
        if basename is None:
            raise Exception('For AIPS data a basename must be specified')
        
        panel_chunk_params['origin'] = 'AIPS'
        _panel_chunk(panel_chunk_params)

    else:
        panel_chunk_params['origin'] = 'astrohack'
        delayed_list = []
        fullname = holog_image+'.image.zarr'
        antennae = os.listdir(fullname)
        count = 0
        for antenna in antennae:
            if 'ant_' in antenna:
                panel_chunk_params['antenna'] = antenna
                
                if ddis is None:
                    ddis = os.listdir(fullname+'/'+antenna)
            
                for ddi in ddis:
                    if 'ddi_' in ddi:
                        info(f"Processing {ddi} for {antenna}")
                        panel_chunk_params['ddi'] = ddi
                        if parallel:
                            delayed_list.append(dask.delayed(_panel_chunk)(dask.delayed(panel_chunk_params)))
                        else:
                            _panel_chunk(panel_chunk_params)
                        count += 1
        if parallel:
            dask.compute(delayed_list)

        if count == 0:
            warning("No data to process")


def _panel_chunk(panel_chunk_params):
    """
    Process a chunk of the holographies, usually a chunk consists of an antenna over a ddi
    Args:
        panel_chunk_params: dictionary of inputs
    """
    if panel_chunk_params['origin'] == 'AIPS':
        telescope = Telescope(panel_chunk_params['telescope'])
        inputxds = xr.open_zarr(panel_chunk_params['holog_image'])
        suffix = ''
        tname = telescope.name.replace(' ', '_')

    else:
        inputxds = _load_image_xds(panel_chunk_params['holog_image'],
                                   panel_chunk_params['antenna'],
                                   panel_chunk_params['ddi'])

        inputxds.attrs['AIPS'] = False

        if inputxds.attrs['telescope_name'] == "ALMA":
            tname = inputxds.attrs['telescope_name']+'_'+inputxds.attrs['ant_name'][0:2]
            telescope = Telescope(tname)
        elif inputxds.attrs['telescope_name'] == "EVLA":
            tname = "VLA"
            telescope = Telescope(tname)
        else:
            raise ValueError('Unsuported telescope {0:s}'.format(inputxds.attrs['telescope_name']))
            
        suffix = '_' + inputxds.attrs['ant_name'] + '/' + panel_chunk_params['ddi']

    surface = AntennaSurface(inputxds, telescope, panel_chunk_params['cutoff'], panel_chunk_params['panel_kind'],
                             panel_margins=panel_chunk_params['panel_margins'])

    surface.compile_panel_points()
    surface.fit_surface()
    surface.correct_surface()

    if panel_chunk_params['basename'] is None:
        basename = panel_chunk_params['outfile'] + '/' + panel_chunk_params['antenna']
    else:
        basename = panel_chunk_params['outfile'] + '/' + panel_chunk_params['basename'] + suffix

    os.makedirs(name=basename, exist_ok=True)

    basename += "/"
    xds = surface.export_xds()
    xds.to_zarr(basename+'xds.zarr', mode='w')
    surface.export_screw_adjustments(basename + "screws.txt", unit=panel_chunk_params['unit'])
    
    if panel_chunk_params['save_mask']:
        surface.plot_surface(filename=basename + "mask.png", mask=True, screws=True)
    
    if panel_chunk_params['save_deviations']:
        surface.plot_surface(filename=basename + "surface.png")
    
    if panel_chunk_params['save_phase']:
        surface.plot_surface(filename=basename + "phase.png", plotphase=True)


def create_phase_model(npix, parameters, wavelength, telescope, cellxy):
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