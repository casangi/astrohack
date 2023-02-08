from astrohack._classes.antenna_surface import AntennaSurface
from astrohack._classes.telescope import Telescope
from astrohack._utils._io import _load_image_xds
import os
import xarray as xr
import dask


def panel(holog_image, aipsdata=False, telescope=None, cutoff=None, panel_kind=None, basename=None, unit='mm',
          save_mask=False, save_deviations=True, save_phase=False, parallel=True):
    panel_chunk_params = {'holog_image': holog_image,
                          'unit': unit,
                          'panel_kind': panel_kind,
                          'cutoff': cutoff,
                          'save_mask': save_mask,
                          'save_deviations': save_deviations,
                          'save_phase': save_phase,
                          'telescope': telescope,
                          'basename': basename
                          }

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
        for antenna in antennae:
            panel_chunk_params['antenna'] = antenna
            ddis = os.listdir(fullname+'/'+antenna)
            for ddi in ddis:
                panel_chunk_params['ddi'] = ddi
                if parallel:
                    delayed_list.append(dask.delayed(_panel_chunk)(dask.delayed(panel_chunk_params)))
                else:
                    _panel_chunk(panel_chunk_params)
        if parallel:
            dask.compute(delayed_list)


def _panel_chunk(panel_chunk_params):

    if panel_chunk_params['origin'] == 'AIPS':
        telescope = Telescope(panel_chunk_params['telescope'])
        inputxds = xr.open_zarr(panel_chunk_params['holog_image'])
        suffix = ''
    else:
        inputxds = _load_image_xds(panel_chunk_params['holog_image'],
                                   panel_chunk_params['antenna'],
                                   panel_chunk_params['ddi'])
        inputxds.attrs['AIPS'] = False
        inputxds.attrs['antenna_name'] = panel_chunk_params['antenna']
        if panel_chunk_params['telescope'] is None:
            tname = inputxds.attrs['telescope_name']+'_'+inputxds.attrs['ant_name'][0:2]
            telescope = Telescope(tname)
        else:
            telescope = Telescope(panel_chunk_params['telescope'])
        suffix = '_' + inputxds.attrs['ant_name'] + '_' + panel_chunk_params['ddi']

    surface = AntennaSurface(inputxds, telescope, panel_chunk_params['cutoff'], panel_chunk_params['panel_kind'])
    surface.compile_panel_points()
    surface.fit_surface()
    surface.correct_surface()

    if panel_chunk_params['basename'] is None:
        basename = telescope.name + suffix
    else:
        basename = panel_chunk_params['basename'] + suffix
    os.makedirs(name=basename, exist_ok=True)
    basename += "/"
    surface.export_screw_adjustments(basename + "screws.txt", unit=panel_chunk_params['unit'])

    if panel_chunk_params['save_mask']:
        surface.plot_surface(filename=basename + "mask.png", mask=True, screws=True)
    if panel_chunk_params['save_deviations']:
        surface.plot_surface(filename=basename + "surface.png")
    if panel_chunk_params['save_phase']:
        surface.plot_surface(filename=basename + "phase.png", plotphase=True)

    ingains, ougains = surface.gains()
    inrms, ourms = surface.get_rms()
    report = open(basename + "report.txt", "w")

    report.write("Gains before correction: Real: {0:7.3} dB, Theoretical: {1:7.3} dB\n".format(*ingains))
    report.write("RMS before correction: {0:7.3} mm\n".format(inrms))
    report.write("\n")
    report.write("Gains after correction: Real: {0:7.3} dB, Theoretical: {1:7.3} dB\n".format(*ougains))
    report.write("RMS after correction: {0:7.3} mm\n".format(ourms))
    report.close()
    return
