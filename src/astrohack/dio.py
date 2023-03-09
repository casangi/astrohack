import os
import dask
import sys

import xarray as xr
import numpy as np

from casacore import tables as ctables

from prettytable import PrettyTable

from astrohack._utils import _system_message as console
from astrohack._utils._io import _load_pnt_dict, _make_ant_pnt_dict
from astrohack._utils._io import _extract_holog_chunk, _open_no_dask_zarr
from astrohack._utils._io import _create_holog_meta_data, _read_data_from_holog_json
from astrohack._utils._io import _read_meta_data

from memory_profiler import profile

def _load_holog_file(holog_file, dask_load=True, load_pnt_dict=True, ant_id=None, holog_dict=None):
    """Loads holog file from disk

    Args:
        holog_name (str): holog file name

    Returns:
        hologfile (nested-dict): {
                            'point.dict':{}, 'ddi':
                                                {'scan':
                                                    {'antenna':
                                                        {
                                                            xarray.DataArray
                                                        }
                                                    }
                                                }
                        }
    """

    if holog_dict is None:
        holog_dict = {}

    if load_pnt_dict == True:
        console.info("Loading pointing dictionary to holog ...")
        holog_dict["pnt_dict"] = _load_pnt_dict(
            file=holog_file, ant_list=None, dask_load=dask_load
        )

    for ddi in os.listdir(holog_file):
        if ddi.isnumeric():
            holog_dict[int(ddi)] = {}
            for scan in os.listdir(os.path.join(holog_file, ddi)):
                if scan.isnumeric():
                    holog_dict[int(ddi)][int(scan)] = {}
                    for ant in os.listdir(os.path.join(holog_file, ddi + "/" + scan)):
                        if ant.isnumeric():
                            mapping_ant_vis_holog_data_name = os.path.join(
                                holog_file, ddi + "/" + scan + "/" + ant
                            )

                            if dask_load:
                                holog_dict[int(ddi)][int(scan)][int(ant)] = xr.open_zarr(
                                    mapping_ant_vis_holog_data_name
                                )
                            else:
                                holog_dict[int(ddi)][int(scan)][
                                    int(ant)
                                ] = _open_no_dask_zarr(mapping_ant_vis_holog_data_name)

    if ant_id == None:
        return holog_dict

    return holog_dict, _read_data_from_holog_json(holog_file=holog_file, holog_dict=holog_dict, ant_id=ant_id)

fp=open('extract_holog.log','w+')
@profile(stream=fp)
def extract_holog(
    ms_name,
    holog_name,
    holog_obs_dict=None,
    data_col="DATA",
    subscan_intent="MIXED",
    parallel=True,
    overwrite=False,
):
    """Extract holography data and create beam maps.
            subscan_intent: 'MIXED' or 'REFERENCE'

    Args:
        ms_name (string): Measurement file name
        holog_name (string): Basename of holog file to create.
        holog_obs_dict (dict): Nested dictionary ordered by ddi:{ scan: { map:[ant names], ref:[ant names] } } }
        data_col (str, optional): Data column from measurement set to acquire. Defaults to 'DATA'.
        subscan_intent (str, optional): Subscan intent, can be MIXED or REFERENCE; MIXED refers to a pointing measurement with half ON(OFF) source. Defaults to 'MIXED'.
        parallel (bool, optional): Boolean for whether to process in parallel. Defaults to True.
        overwrite (bool, optional): Boolean for whether to overwrite current holography file.
    """

    holog_file = "{base}.{suffix}".format(base=holog_name, suffix="holog.zarr")

    if os.path.exists(holog_file) is True and overwrite is False:
        console.error(
            "[_create_holog_file] holog file {file} exists. To overwite set the overwrite=True option in extract_holog or remove current file.".format(
                file=holog_file
            )
        )
        raise FileExistsError
    else:
        console.warning(
            "[extract_holog] Warning, current holography files will be overwritten."
        )

    pnt_name = "{base}.{pointing}".format(base=holog_name, pointing="point.zarr")

    #pnt_dict = _load_pnt_dict(pnt_name)
    pnt_dict = _make_ant_pnt_dict(ms_name, pnt_name, parallel=parallel)
    
    if holog_obs_dict is None:
        ant_names_list = []
        #Create mapping antennas
        holog_obs_dict = {}
        for ant_id,pnt_xds in pnt_dict.items():
            ant_name = pnt_xds.attrs['ant_name']
            mapping_scans = pnt_xds.attrs['mapping_scans']
            ant_names_list.append(ant_name)
            for ddi,scans in  mapping_scans.items():
                ddi = int(ddi)
                for s in scans:
                    try:
                        holog_obs_dict[ddi][s]['map'].append(ant_name)
                    except:
                        holog_obs_dict.setdefault(ddi,{})[s] = {'map':[ant_name]}#dict(zip(scans, [ant_name]*len(scans)))
       
       
        #Create reference antennas
        
        ant_names_set = set(ant_names_list)
        for ddi,scan in holog_obs_dict.items():
            for scan_id,ant_types in scan.items():
                holog_obs_dict[ddi][scan_id]['ref'] = ant_names_set - set(holog_obs_dict[ddi][scan_id]['map'])
            
    print(holog_obs_dict)


    ######## Get Spectral Windows ########

    # nomodify=True when using CASA tables.
    # print(os.path.join(ms_name,"DATA_DESCRIPTION"))
    console.info(
        "Opening measurement file {ms}".format(
            ms=os.path.join(ms_name, "DATA_DESCRIPTION")
        )
    )

    ctb = ctables.table(
        os.path.join(ms_name, "DATA_DESCRIPTION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    ddi_spw = ctb.getcol("SPECTRAL_WINDOW_ID")
    ddpol_indexol = ctb.getcol("POLARIZATION_ID")
    ddi = np.arange(len(ddi_spw))
    ctb.close()

    ######## Get Antenna IDs and Names ########
    ctb = ctables.table(
        os.path.join(ms_name, "ANTENNA"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant_name = ctb.getcol("NAME")
    ant_id = np.arange(len(ant_name))

    ctb.close()

    ######## Get Scan and Subscan IDs ########
    # SDM Tables Short Description (https://drive.google.com/file/d/16a3g0GQxgcO7N_ZabfdtexQ8r2jRbYIS/view)
    # 2.54 ScanIntent (p. 150)
    # MAP ANTENNA SURFACE : Holography calibration scan

    # 2.61 SubscanIntent (p. 152)
    # MIXED : Pointing measurement, some antennas are on -ource, some off-source
    # REFERENCE : reference measurement (used for boresight in holography).
    # Undefined : ?

    ctb = ctables.table(
        os.path.join(ms_name, "STATE"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    # scan intent (with subscan intent) is stored in the OBS_MODE column of the STATE subtable.
    obs_modes = ctb.getcol("OBS_MODE")
    ctb.close()

    scan_intent = "MAP_ANTENNA_SURFACE"
    state_ids = []

    for i, mode in enumerate(obs_modes):
        if (scan_intent in mode) and (subscan_intent in mode):
            state_ids.append(i)

    spw_ctb = ctables.table(
        os.path.join(ms_name, "SPECTRAL_WINDOW"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    pol_ctb = ctables.table(
        os.path.join(ms_name, "POLARIZATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    obs_ctb = ctables.table(
        os.path.join(ms_name, "OBSERVATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    delayed_list = []
    for ddi in holog_obs_dict:

        spw_setup_id = ddi_spw[ddi]
        pol_setup_id = ddpol_indexol[ddi]

        extract_holog_params = {
            "ms_name": ms_name,
            "holog_name": holog_name,
            "pnt_name": pnt_name,
            "ddi": ddi,
            "data_col": data_col,
            "chan_setup": {},
            "pol_setup": {},
            "telescope_name": "",
            "overwrite": overwrite,
        }

        extract_holog_params["chan_setup"]["chan_freq"] = spw_ctb.getcol("CHAN_FREQ", startrow=spw_setup_id, nrow=1)[0, :]
        extract_holog_params["chan_setup"]["chan_width"] = spw_ctb.getcol("CHAN_WIDTH", startrow=spw_setup_id, nrow=1)[0, :]
        extract_holog_params["chan_setup"]["eff_bw"] = spw_ctb.getcol("EFFECTIVE_BW", startrow=spw_setup_id, nrow=1)[0, :]
        extract_holog_params["chan_setup"]["ref_freq"] = spw_ctb.getcol("REF_FREQUENCY", startrow=spw_setup_id, nrow=1)[0]
        extract_holog_params["chan_setup"]["total_bw"] = spw_ctb.getcol("TOTAL_BANDWIDTH", startrow=spw_setup_id, nrow=1)[0]

        extract_holog_params["pol_setup"]["pol"] = pol_ctb.getcol("CORR_TYPE", startrow=spw_setup_id, nrow=1)[0, :]
        
        extract_holog_params["telescope_name"] = obs_ctb.getcol("TELESCOPE_NAME")[0]

        for scan in holog_obs_dict[ddi].keys():
            console.info("Processing ddi: {ddi}, scan: {scan}".format(ddi=ddi, scan=scan))

            map_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]["map"]))[0]
            ref_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]["ref"]))[0]

            extract_holog_params["map_ant_ids"] = map_ant_ids
            extract_holog_params["ref_ant_ids"] = ref_ant_ids
            extract_holog_params["sel_state_ids"] = state_ids
            extract_holog_params["scan"] = scan

            if parallel:
                delayed_list.append(
                    dask.delayed(_extract_holog_chunk)(
                        dask.delayed(extract_holog_params)
                    )
                )
            else:
                _extract_holog_chunk(extract_holog_params)

    spw_ctb.close()
    pol_ctb.close()

    if parallel:
        dask.compute(delayed_list)

    extract_holog_params["holog_obs_dict"] = {}

    for id in ant_id:
        extract_holog_params["holog_obs_dict"][str(id)] = ant_name[id]

    holog_file = "{base}.{suffix}".format(base=extract_holog_params["holog_name"], suffix="holog.zarr")

    holog_dict = _load_holog_file(holog_file=holog_file, dask_load=True, load_pnt_dict=False)

    _create_holog_meta_data(holog_file=holog_file, holog_dict=holog_dict, holog_params=extract_holog_params)

class AstrohackDataFile:
    def __init__(self, file_stem, path='./'):
                        
        self._image_path = None
        self._holog_path = None

        self.holog = None
        self.image = None
            
        self._verify_holog_files(file_stem, path)
            

    def _verify_holog_files(self, file_stem, path):
        console.info("Verifying {stem}.* files in path={path} ...".format(stem=file_stem, path=path))

        file_path = "{path}/{stem}.holog.zarr".format(path=path, stem=file_stem)
            
        if os.path.isdir(file_path):
            console.info("Found {stem}.holog.zarr directory ...".format(stem=file_stem))
            self._holog_path = file_path
            self.holog = AstrohackHologFile(file_path)
                

        file_path = "{path}/{stem}.image.zarr".format(path=path, stem=file_stem)

        if os.path.isdir(file_path):
            console.info("Found {stem}.image.zarr directory ...".format(stem=file_stem))
            self._image_path = file_path
            self.image = AstrohackImageFile(file_path)

class AstrohackImageFile(dict):
    """
        Data class for holography image data.
    """
    def __init__(self, file):
        super().__init__()

        self.file = file
        self._open = False

    def __getitem__(self, key):
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        return super().__setitem__(key, value)
        
    def is_open(self):
        return self._open

    def open(self, file=None):
        """ Open hologgraphy file.

        Args:self =_
            file (str, optional): Path to holography file. Defaults to None.

        Returns:
            bool: bool describing whether the file was opened properly
        """
        from astrohack._utils._io import _load_image_xds

        if file is None:
            file = self.file

        ant_list =  [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]
        
        try:
            for ant in ant_list:
                ddi_list =  [dir_name for dir_name in os.listdir(file + "/" + str(ant)) if os.path.isdir(file + "/" + str(ant))]
                self[int(ant)] = {}
                for ddi in ddi_list:
                    self[int(ant)][int(ddi)] = xr.open_zarr("{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi) )

            self._open = True

        except Exception as e:
            console.error("[AstroHackImageFile.open()]: {}".format(e))
            self._open = False

        return self._open

    def summary(self):
        """
           Prints summary table of holog image file. 
        """

        table = PrettyTable()
        table.field_names = ["antenna", "ddi"]
        table.align = "l"
        
        for ant in self.keys():
            table.add_row([ant, list(self[int(ant)].keys())])
        
        print(table)


    def select(self, ant=None, ddi=None, polar=False):
        """Select data on the basis of ddi, scan, ant. This is a convenience function.

        Args:
            ddi (int, optional): Data description ID. Defaults to None.
            ant (int, optional): Antenna ID. Defaults to None.

        Returns:
            xarray.Dataset: xarray dataset of corresponding ddi, scan, antenna ID.
        """
        if ant is None and ddi is None:
            console.info("No selections made ...")
            return self
        else:
            if polar:
                return self[ant][ddi].apply(np.absolute), self[ant][ddi].apply(np.angle, deg=True)

            return self[ant][ddi]

class AstrohackHologFile(dict):
    """
        Data Class to interact ith holography imaging data.
    """
    def __init__(self, file):
        super().__init__()
        
        self.file = file
        self._meta_data = None
        self._open = False


    def __getitem__(self, key):
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    def is_open(self):
        return self._open

    def open(self, file=None, dask_load=False):
        """ Open hologgraphy file.

        Args:self =_
            file (str, optional): Path to holography file. Defaults to None.
            dask_load (bool, optional): If True the file is loaded with Dask. Defaults to False.

        Returns:
            bool: bool describing whether the file was opened properly
        """

        if file is None:
            file = self.file

        self._meta_data = _read_meta_data(holog_file=file)

        try:
            _load_holog_file(holog_file=file, dask_load=dask_load, load_pnt_dict=False, holog_dict=self)
            self._open = True

        except Exception as e:
            console.error("[AstrohackHologFile]: {}".format(e))
            self._open = False
        
        return self._open

    def summary(self):
        """
            Prints summary table of holog file.
        """

        table = PrettyTable()
        table.field_names = ["ddi", "scan", "antenna"]
        table.align = "l"
        
        for ddi in self.keys():
            for scan in self[int(ddi)].keys():
                table.add_row([ddi, scan, list(self[int(ddi)][int(scan)].keys())])
        
        print(table)

    def select(self, ddi=None, scan=None, ant=None):
        """ Select data on the basis of ddi, scan, ant. This is a convenience function.

        Args:
            ddi (int, optional): Data description ID. Defaults to None.
            scan (int, optional): Scan number. Defaults to None.
            ant (int, optional): Antenna ID. Defaults to None.

        Returns:
            xarray.Dataset: xarray dataset of corresponding ddi, scan, antenna ID.
        """
        if ant is None or ddi is None or scan is None:
            console.info("No selections made ...")
            return self
        else:
            return self[ddi][scan][ant]

    @property
    def meta_data(self):
        """ Holog file meta data.

        Returns:
            JSON: JSON file of holography meta data.
        """

        return self._meta_data
