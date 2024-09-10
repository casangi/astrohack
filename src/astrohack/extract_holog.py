import copy
import json
import os
import pathlib
import pickle
import shutil
import math
import multiprocessing

import toolviper.utils.parameter
import dask
import astrohack
import psutil

import numpy as np
import toolviper.utils.logger as logger

from astropy.time import Time
from casacore import tables as ctables
from rich.console import Console
from rich.table import Table

from astrohack.utils.constants import pol_str

from astrohack.utils.file import overwrite_file
from astrohack.utils.file import load_holog_file
from astrohack.utils.file import load_point_file
from astrohack.utils.data import write_meta_data
from astrohack.core.extract_holog import create_holog_meta_data
from astrohack.core.extract_holog import create_holog_obs_dict
from astrohack.core.extract_holog import process_extract_holog_chunk
from astrohack.utils.tools import get_valid_state_ids
from astrohack.utils.text import get_default_file_name
from astrohack.utils.text import NumpyEncoder
from astrohack.mds import AstrohackHologFile
from astrohack.mds import AstrohackPointFile
from astrohack.extract_pointing import extract_pointing

from typing import Union, List, NewType, Dict, Any, Tuple

JSON = NewType("JSON", Dict[str, Any])
KWARGS = NewType("KWARGS", Union[Dict[str, str], Dict[str, int]])


class HologObsDict(dict):
    """
      ddi --> map --> ant, scan
                      |
                      o--> map: [reference, ...]
    """

    def __init__(self, obj: JSON = None):
        if obj is None:
            super().__init__()
        else:
            super().__init__(obj)

    def __getitem__(self, key: str):
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(key, value)

    @classmethod
    def from_file(cls, filepath):
        if filepath.endswith(".holog.zarr"):
            filepath = str(pathlib.Path(filepath).resolve().joinpath("holog_obs_dict.json"))

        try:
            with open(filepath, "r") as file:
                obj = json.load(file)

                return HologObsDict(obj)

        except FileNotFoundError:
            logger.error(f"File {filepath} not found")

    def print(self, style: str = "static"):
        if style == "dynamic":
            return astrohack.dio.inspect_holog_obs_dict(self, style="dynamic")

        else:
            return astrohack.dio.inspect_holog_obs_dict(self, style="static")

    def select(self, key: str, value: any, inplace: bool = False, **kwargs: KWARGS) -> object:

        if inplace:
            obs_dict = self

        else:
            obs_dict = HologObsDict(copy.deepcopy(self))

        if key == "ddi":
            return self._select_ddi(value, obs_dict=obs_dict)

        elif key == "map":
            return self._select_map(value, obs_dict=obs_dict)

        elif key == "antenna":
            return self._select_antenna(value, obs_dict=obs_dict)

        elif key == "scan":
            return self._select_scan(value, obs_dict=obs_dict)

        elif key == "baseline":
            if "reference" in kwargs.keys():
                return self._select_baseline(
                    value,
                    n_baselines=None,
                    reference=kwargs["reference"],
                    obs_dict=obs_dict
                )

            elif "n_baselines" in kwargs.keys():
                return self._select_baseline(
                    value,
                    n_baselines=kwargs["n_baselines"],
                    reference=None,
                    obs_dict=obs_dict
                )

            else:
                logger.error("Must specify a list of reference antennas for this option.")
        else:
            logger.error("Valid key not found: {key}".format(key=key))
            return {}

    @staticmethod
    def get_nearest_baselines(antenna: str, n_baselines: int = None, path_to_matrix: str = None) -> object:
        import pandas as pd

        if path_to_matrix is None:
            path_to_matrix = str(pathlib.Path.cwd().joinpath(".baseline_distance_matrix.csv"))

        if not pathlib.Path(path_to_matrix).exists():
            logger.error("Unable to find baseline distance matrix in: {path}".format(path=path_to_matrix))

        df_matrix = pd.read_csv(path_to_matrix, sep="\t", index_col=0)

        # Skip the first index because it is a self distance
        if n_baselines is None:
            return df_matrix[antenna].sort_values(ascending=True).index[1:].values.tolist()

        return df_matrix[antenna].sort_values(ascending=True).index[1:n_baselines].values.tolist()

    @staticmethod
    def _select_ddi(value: Union[int, List[int]], obs_dict: object) -> object:
        convert = lambda x: "ddi_" + str(x)

        if not isinstance(value, list):
            value = [value]

        value = list(map(convert, value))
        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            if ddi not in value:
                obs_dict.pop(ddi)

        return obs_dict

    @staticmethod
    def _select_map(value: Union[int, List[int]], obs_dict: object) -> object:
        convert = lambda x: "map_" + str(x)

        if not isinstance(value, list):
            value = [value]

        value = list(map(convert, value))
        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            map_list = list(obs_dict[ddi].keys())
            for mp in map_list:
                if mp not in value:
                    obs_dict[ddi].pop(mp)

        return obs_dict

    @staticmethod
    def _select_antenna(value: Union[str, List[str]], obs_dict: object) -> object:
        if not isinstance(value, list):
            value = [value]

        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            map_list = list(obs_dict[ddi].keys())
            for mp in map_list:
                ant_list = list(obs_dict[ddi][mp]["ant"].keys())
                for ant in ant_list:
                    if ant not in value:
                        obs_dict[ddi][mp]["ant"].pop(ant)

        return obs_dict

    @staticmethod
    def _select_scan(value: Union[int, List[int]], obs_dict: object) -> object:
        if not isinstance(value, list):
            value = [value]

        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            map_list = list(obs_dict[ddi].keys())
            for mp in map_list:
                obs_dict[ddi][mp]["scans"] = value

        return obs_dict

    @staticmethod
    def _select_baseline(
            value: str,
            n_baselines: int,
            obs_dict: object,
            reference: Union[str, List[int]] = None
    ) -> object:
        if reference is not None:
            if not isinstance(reference, list):
                reference = [reference]

        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            map_list = list(obs_dict[ddi].keys())
            for mp in map_list:
                ant_list = list(obs_dict[ddi][mp]["ant"].keys())
                for ant in ant_list:
                    if ant not in value:
                        obs_dict[ddi][mp]["ant"].pop(ant)
                        continue

                    if reference is None and n_baselines is not None:
                        reference_antennas = obs_dict[ddi][mp]["ant"][ant]

                        if n_baselines > len(reference_antennas):
                            n_baselines = len(reference_antennas)

                        sorted_antennas = np.array(obs_dict.get_nearest_baselines(antenna=ant))

                        values, i, j = np.intersect1d(reference_antennas, sorted_antennas, return_indices=True)
                        index = np.sort(j)

                        obs_dict[ddi][mp]["ant"][ant] = sorted_antennas[index][:n_baselines]

                    else:
                        obs_dict[ddi][mp]["ant"][ant] = reference

        return obs_dict


@toolviper.utils.parameter.validate(
    add_data_type=HologObsDict
)
def extract_holog(
        ms_name: str,
        point_name: str,
        holog_name: str = None,
        holog_obs_dict: HologObsDict = None,
        ddi: Union[int, List[int], str] = 'all',
        baseline_average_distance: Union[float, str] = 'all',
        baseline_average_nearest: Union[float, str] = 'all',
        data_column: str = "CORRECTED_DATA",
        time_smoothing_interval: float = None,
        parallel: bool = False,
        overwrite: bool = False,
) -> Union[AstrohackHologFile, None]:
    """
    Extract holography and optionally pointing data, from measurement set. Creates holography output file.

    :param ms_name: Name of input measurement file name.
    :type ms_name: str

    :param point_name: Name of *<point_name>.point.zarr* file to use. This is must be provided.
    :type holog_name: str

    :param holog_name: Name of *<holog_name>.holog.zarr* file to create. Defaults to measurement set name with
    *holog.zarr* extension.
    :type holog_name: str, optional

    :param holog_obs_dict: The *holog_obs_dict* describes which scan and antenna data to extract from the measurement \
    set. As detailed below, this compound dictionary also includes important metadata needed for preprocessing and \
    extraction of the holography data from the measurement set. If not specified holog_obs_dict will be generated. \
    For auto generation of the holog_obs_dict the assumption is made that the same antenna beam is not mapped twice in \
    a row (alternating sets of antennas is fine). If the holog_obs_dict is specified, the ddi input is ignored. The \
    user can self generate this dictionary using `generate_holog_obs_dict`.
    :type holog_obs_dict: dict, optional

    :param ddi:  DDI(s) that should be extracted from the measurement set. Defaults to all DDI's in the ms.
    :type ddi: int numpy.ndarray | int list, optional

    :param baseline_average_distance: To increase the signal-to-noise for a mapping antenna multiple reference \
    antennas can be used. The baseline_average_distance is the acceptable distance (in meters) between a mapping \
    antenna and a reference antenna. The baseline_average_distance is only used if the holog_obs_dict is not \
    specified. If no distance is specified all reference antennas will be used. baseline_average_distance and \
    baseline_average_nearest can not be used together.
    :type baseline_average_distance: float, optional

    :param baseline_average_nearest: To increase the signal-to-noise for a mapping antenna multiple reference antennas \
    can be used. The baseline_average_nearest is the number of nearest reference antennas to use. The \
    baseline_average_nearest is only used if the holog_obs_dict is not specified.  baseline_average_distance and \
    baseline_average_nearest can not be used together.
    :type baseline_average_nearest: int, optional

    :param data_column: Determines the data column to pull from the measurement set. Defaults to "CORRECTED_DATA".
    :type data_column: str, optional, ex. DATA, CORRECTED_DATA

    :param time_smoothing_interval: Determines the time smoothing interval, set to the integration time when None.
    :type time_smoothing_interval: float, optional

    :param parallel: Boolean for whether to process in parallel, defaults to False.
    :type parallel: bool, optional

    :param overwrite: Boolean for whether to overwrite current holog.zarr and point.zarr files, defaults to False.
    :type overwrite: bool, optional

    :return: Holography holog object.
    :rtype: AstrohackHologFile

    .. _Description:

    **AstrohackHologFile**

    Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth,
    `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for
    each file. An outline of the holog object structure is show below:

    .. parsed-literal::
        holog_mds = 
        {
            ddi_0:{
                map_0:{
                 ant_0: holog_ds,
                          ⋮
                 ant_n: holog_ds
                },
                ⋮
                map_p: …
            },
            ⋮
            ddi_m: …
        }

    **Example Usage** In this case the pointing file has already been created. In addition, the appropriate
    data_column value nees to be set for the type of measurement set data you are extracting.

    .. parsed-literal::
        from astrohack.extract_holog import extract_holog

        holog_mds = extract_holog(
            ms_name="astrohack_observation.ms",
            point_name="astrohack_observation.point.ms",
            holog_name="astrohack_observation.holog.ms",
            data_column='CORRECTED_DATA',
            parallel=True,
            overwrite=True
        )

    **Additional Information**

        This function extracts the holography related information from the given measurement file. The data is
        restructured into an astrohack file format and saved into a file in the form of *<holog_name>.holog.zarr*.
        The extension *.holog.zarr* is used for all holography files. In addition, the pointing information is
        recorded into a holography file of format *<pointing_name>.point.zarr*. The extension *.point.zarr* is used
        for all holography pointing files.

        **holog_obs_dict[holog_mapping_id] (dict):** *holog_mapping_id* is a unique, arbitrary, user-defined integer
        assigned to the data that describes a single complete mapping of the beam.
        
        .. rubric:: This is needed for two reasons:
        * A complete mapping of the beam can be done over more than one scan (for example the VLA data). 
        * A measurement set can contain more than one mapping of the beam (for example the ALMA data).
    
        **holog_obs_dict[holog_mapping_id][scans] (int | numpy.ndarray | list):**
        All the scans in the measurement set the *holog_mapping_id*.
    
        **holog_obs_dict[holog_mapping_id][ant] (dict):** The dictionary keys are the mapping antenna names and the
        values a list of the reference antennas. See example below.

        The below example shows how the *holog_obs_description* dictionary should be laid out. For each
        *holog_mapping_id* the relevant scans and antennas must be provided. For the `ant` key, an entry is required
        for each mapping antenna and the accompanying reference antenna(s).
    
        .. parsed-literal::
            holog_obs_description = {
                'map_0' :{
                    'scans':[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                    'ant':{
                        'DA44':[
                            'DV02', 'DV03', 'DV04', 
                            'DV11', 'DV12', 'DV13', 
                            'DV14', 'DV15', 'DV16', 
                            'DV17', 'DV18', 'DV19', 
                            'DV20', 'DV21', 'DV22', 
                            'DV23', 'DV24', 'DV25'
                        ]
                    }
                }
            }

    """
    # Doing this here allows it to get captured by locals()
    if holog_name is None:
        holog_name = get_default_file_name(input_file=ms_name, output_type=".holog.zarr")

    extract_holog_params = locals()

    input_pars = extract_holog_params.copy()

    assert pathlib.Path(extract_holog_params['ms_name']).exists() is True, (
        logger.error(f'File {extract_holog_params["ms_name"]} does not exists.')
    )

    overwrite_file(extract_holog_params['holog_name'], extract_holog_params['overwrite'])

    try:
        pnt_dict = load_point_file(extract_holog_params['point_name'])

    except Exception as error:
        logger.error('Error loading {name}. - {error}'.format(name=extract_holog_params["point_name"], error=error))

        return None

    # Get spectral windows
    ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "DATA_DESCRIPTION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ddi_spw = ctb.getcol("SPECTRAL_WINDOW_ID")
    ddpol_indexol = ctb.getcol("POLARIZATION_ID")
    ms_ddi = np.arange(len(ddi_spw))
    ctb.close()

    # Get antenna IDs and names
    ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "ANTENNA"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant_names = np.array(ctb.getcol("NAME"))
    ant_id = np.arange(len(ant_names))
    ant_pos = ctb.getcol("POSITION")

    ctb.close()

    # Get antenna IDs in the main table
    ctb = ctables.table(
        extract_holog_params['ms_name'],
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant1 = np.unique(ctb.getcol("ANTENNA1"))
    ant2 = np.unique(ctb.getcol("ANTENNA2"))
    ant_id_main = np.unique(np.append(ant1, ant2))

    ant_names_main = ant_names[ant_id_main]
    ctb.close()

    # Create holog_obs_dict or modify user supplied holog_obs_dict.
    ddi = extract_holog_params['ddi']
    if isinstance(ddi, int):
        ddi = [ddi]

    # Create holog_obs_dict if not specified
    if holog_obs_dict is None:
        holog_obs_dict = create_holog_obs_dict(
            pnt_dict,
            extract_holog_params['baseline_average_distance'],
            extract_holog_params['baseline_average_nearest'],
            ant_names,
            ant_pos,
            ant_names_main
        )

        # From the generated holog_obs_dict subselect user supplied ddis.
        if ddi != 'all':
            holog_obs_dict_keys = list(holog_obs_dict.keys())
            for ddi_key in holog_obs_dict_keys:
                if 'ddi' in ddi_key:
                    ddi_id = int(ddi_key.replace('ddi_', ''))
                    if ddi_id not in ddi:
                        del holog_obs_dict[ddi_key]


    ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "STATE"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    # Scan intent (with subscan intent) is stored in the OBS_MODE column of the STATE sub-table.
    obs_modes = ctb.getcol("OBS_MODE")
    ctb.close()

    state_ids = get_valid_state_ids(obs_modes)

    spw_ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "SPECTRAL_WINDOW"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    pol_ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "POLARIZATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    obs_ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "OBSERVATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    telescope_name = obs_ctb.getcol("TELESCOPE_NAME")[0]
    start_time_unix = obs_ctb.getcol('TIME_RANGE')[0][0] - 3506716800.0
    time = Time(start_time_unix, format='unix').jyear

    # If we have an EVLA run from before 2023 the pointing table needs to be fixed.
    if telescope_name == "EVLA" and time < 2023:

        # Convert from casa epoch to unix time
        his_ctb = ctables.table(
            os.path.join(extract_holog_params['ms_name'], "HISTORY"),
            readonly=True,
            lockoptions={"option": "usernoread"},
            ack=False,
        )

        if "pnt_tbl:fixed" not in his_ctb.getcol("MESSAGE"):
            logger.error(
                "Pointing table not corrected, users should apply function astrohack.dio.fix_pointing_table() to "
                "remedy this.")

            return None

        his_ctb.close()

    count = 0
    delayed_list = []

    for ddi_name in holog_obs_dict.keys():
        ddi = int(ddi_name.replace('ddi_', ''))
        spw_setup_id = ddi_spw[ddi]
        pol_setup_id = ddpol_indexol[ddi]

        extract_holog_params["ddi"] = ddi
        extract_holog_params["chan_setup"] = {}
        extract_holog_params["pol_setup"] = {}
        extract_holog_params["chan_setup"]["chan_freq"] = \
            spw_ctb.getcol("CHAN_FREQ", startrow=spw_setup_id, nrow=1)[0, :]

        extract_holog_params["chan_setup"]["chan_width"] = \
            spw_ctb.getcol("CHAN_WIDTH", startrow=spw_setup_id, nrow=1)[0, :]

        extract_holog_params["chan_setup"]["eff_bw"] = \
            spw_ctb.getcol("EFFECTIVE_BW", startrow=spw_setup_id, nrow=1)[0, :]

        extract_holog_params["chan_setup"]["ref_freq"] = \
            spw_ctb.getcol("REF_FREQUENCY", startrow=spw_setup_id, nrow=1)[0]

        extract_holog_params["chan_setup"]["total_bw"] = \
            spw_ctb.getcol("TOTAL_BANDWIDTH", startrow=spw_setup_id, nrow=1)[0]

        extract_holog_params["pol_setup"]["pol"] = pol_str[
            pol_ctb.getcol("CORR_TYPE", startrow=pol_setup_id, nrow=1)[0, :]]

        extract_holog_params["telescope_name"] = obs_ctb.getcol("TELESCOPE_NAME")[0]

        # Loop over all beam_scan_ids, a beam_scan_id can consist of more than one scan in a measurement set (this is
        # the case for the VLA pointed mosaics).
        for holog_map_key in holog_obs_dict[ddi_name].keys():

            if 'map' in holog_map_key:
                scans = holog_obs_dict[ddi_name][holog_map_key]["scans"]
                if len(scans) > 1:
                    logger.info("Processing ddi: {ddi}, scans: [{min} ... {max}]".format(
                        ddi=ddi, min=scans[0], max=scans[-1]
                    ))
                else:
                    logger.info("Processing ddi: {ddi}, scan: {scan}".format(
                        ddi=ddi, scan=scans
                    ))

                if len(list(holog_obs_dict[ddi_name][holog_map_key]['ant'].keys())) != 0:
                    map_ant_list = []
                    ref_ant_per_map_ant_list = []

                    map_ant_name_list = []
                    ref_ant_per_map_ant_name_list = []
                    for map_ant_str in holog_obs_dict[ddi_name][holog_map_key]['ant'].keys():
                        ref_ant_ids = np.array(_convert_ant_name_to_id(ant_names, list(
                            holog_obs_dict[ddi_name][holog_map_key]['ant'][map_ant_str])))

                        map_ant_id = _convert_ant_name_to_id(ant_names, map_ant_str)[0]

                        ref_ant_per_map_ant_list.append(ref_ant_ids)
                        map_ant_list.append(map_ant_id)

                        ref_ant_per_map_ant_name_list.append(
                            list(holog_obs_dict[ddi_name][holog_map_key]['ant'][map_ant_str]))
                        map_ant_name_list.append(map_ant_str)

                    extract_holog_params["ref_ant_per_map_ant_tuple"] = tuple(ref_ant_per_map_ant_list)
                    extract_holog_params["map_ant_tuple"] = tuple(map_ant_list)

                    extract_holog_params["ref_ant_per_map_ant_name_tuple"] = tuple(ref_ant_per_map_ant_name_list)
                    extract_holog_params["map_ant_name_tuple"] = tuple(map_ant_name_list)

                    extract_holog_params["scans"] = scans
                    extract_holog_params["sel_state_ids"] = state_ids
                    extract_holog_params["holog_map_key"] = holog_map_key
                    extract_holog_params["ant_names"] = ant_names

                    if parallel:
                        delayed_list.append(
                            dask.delayed(process_extract_holog_chunk)(
                                dask.delayed(extract_holog_params)
                            )
                        )
                    else:
                        process_extract_holog_chunk(extract_holog_params)

                    count += 1

                else:
                    logger.warning("DDI " + str(ddi) + " has no holography data to extract.")

    spw_ctb.close()
    pol_ctb.close()
    obs_ctb.close()

    if parallel:
        dask.compute(delayed_list)

    if count > 0:
        logger.info("Finished processing")

        holog_dict = load_holog_file(
            file=extract_holog_params["holog_name"],
            dask_load=True,
            load_pnt_dict=False
        )

        extract_holog_params['telescope_name'] = telescope_name

        meta_data = create_holog_meta_data(
            holog_file=extract_holog_params['holog_name'],
            holog_dict=holog_dict,
            input_params=extract_holog_params.copy()
        )

        holog_attr_file = "{name}/{ext}".format(name=extract_holog_params['holog_name'], ext=".holog_attr")
        write_meta_data(holog_attr_file, meta_data)

        holog_attr_file = "{name}/{ext}".format(name=extract_holog_params['holog_name'], ext=".holog_input")
        write_meta_data(holog_attr_file, input_pars)

        with open(f"{extract_holog_params['holog_name']}/holog_obs_dict.json", "w") as outfile:
            json.dump(holog_obs_dict, outfile, cls=NumpyEncoder)

        holog_mds = AstrohackHologFile(extract_holog_params['holog_name'])
        holog_mds.open()

        return holog_mds

    else:
        logger.warning("No data to process")
        return None


def generate_holog_obs_dict(
        ms_name: str,
        point_name: str,
        baseline_average_distance: str = 'all',
        baseline_average_nearest: str = 'all',
        write=True,
        parallel: bool = False
) -> HologObsDict:
    """
    Generate holography observation dictionary, from measurement set..

    :param ms_name: Name of input measurement file name.
    :type ms_name: str

    :param baseline_average_distance: To increase the signal-to-noise for a mapping antenna multiple reference
    antennas can be used. The baseline_average_distance is the acceptable distance between a mapping antenna and a
    reference antenna. The baseline_average_distance is only used if the holog_obs_dict is not specified. If no
    distance is specified all reference antennas will be used. baseline_average_distance and baseline_average_nearest
    can not be used together.
    :type baseline_average_distance: float, optional

    :param baseline_average_nearest: To increase the signal-to-noise for a mapping antenna multiple reference antennas
    can be used. The baseline_average_nearest is the number of nearest reference antennas to use. The
    baseline_average_nearest is only used if the holog_obs_dict is not specified.  baseline_average_distance and
    baseline_average_nearest can not be used together.
    :type baseline_average_nearest: int, optional

    :param write: Write file flag.
    :type point_name: bool, optional

    :param point_name: Name of *<point_name>.point.zarr* file to use. 
    :type point_name: str, optional

    :param parallel: Boolean for whether to process in parallel. Defaults to False
    :type parallel: bool, optional

    :return: holog observation dictionary
    :rtype: json

    .. _Description:

    **AstrohackHologFile**

    Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth,
    `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for
    each file. An outline of the holog object structure is show below:

    .. parsed-literal::
        holog_mds = 
        {
            ddi_0:{
                map_0:{
                 ant_0: holog_ds,
                          ⋮
                 ant_n: holog_ds
                },
                ⋮
                map_p: …
            },
            ⋮
            ddi_m: …
        }

    **Example Usage**
    In this case the pointing file has already been created. 

    .. parsed-literal::
        from astrohack.extract_holog import generate_holog_obs_dict

        holog_obs_obj = generate_holog_obs_dict(
            ms_name="astrohack_observation.ms",
            point_name="astrohack_observation.point.zarr"
        )

    **Additional Information**

        **holog_obs_dict[holog_mapping_id] (dict):** *holog_mapping_id* is a unique, arbitrary, user-defined integer
        assigned to the data that describes a single complete mapping of the beam.
        
        .. rubric:: This is needed for two reasons:
        * A complete mapping of the beam can be done over more than one scan (for example the VLA data). 
        * A measurement set can contain more than one mapping of the beam (for example the ALMA data).
    
        **holog_obs_dict[holog_mapping_id][scans] (int | numpy.ndarray | list):**
        All the scans in the measurement set the *holog_mapping_id*.
    
        **holog_obs_dict[holog_mapping_id][ant] (dict):** The dictionary keys are the mapping antenna names and the
        values a list of the reference antennas. See example below.

        The below example shows how the *holog_obs_description* dictionary should be laid out. For each
        *holog_mapping_id* the relevant scans and antennas must be provided. For the `ant` key, an entry is required
        for each mapping antenna and the accompanying reference antenna(s).
    
        .. parsed-literal::
            holog_obs_description = {
                'map_0' :{
                    'scans':[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                    'ant':{
                        'DA44':[
                            'DV02', 'DV03', 'DV04', 
                            'DV11', 'DV12', 'DV13', 
                            'DV14', 'DV15', 'DV16', 
                            'DV17', 'DV18', 'DV19', 
                            'DV20', 'DV21', 'DV22', 
                            'DV23', 'DV24', 'DV25'
                        ]
                    }
                }
            }

    """
    extract_holog_params = locals()

    assert pathlib.Path(ms_name).exists() is True, logger.error(f'File {ms_name} does not exists.')
    assert pathlib.Path(point_name).exists() is True, logger.error(f'File {point_name} does not exists.')

    # Get antenna IDs and names
    ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "ANTENNA"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant_names = np.array(ctb.getcol("NAME"))
    ant_id = np.arange(len(ant_names))
    ant_pos = ctb.getcol("POSITION")

    ctb.close()

    # Get antenna IDs that are in the main table
    ctb = ctables.table(
        extract_holog_params['ms_name'],
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant1 = np.unique(ctb.getcol("ANTENNA1"))
    ant2 = np.unique(ctb.getcol("ANTENNA2"))
    ant_id_main = np.unique(np.append(ant1, ant2))

    ant_names_main = ant_names[ant_id_main]
    ctb.close()

    pnt_mds = AstrohackPointFile(extract_holog_params['point_name'])
    pnt_mds.open()

    holog_obs_dict = create_holog_obs_dict(
        pnt_mds,
        extract_holog_params['baseline_average_distance'],
        extract_holog_params['baseline_average_nearest'],
        ant_names,
        ant_pos,
        ant_names_main,
        write_distance_matrix=True
    )

    encoded_obj = json.dumps(holog_obs_dict, cls=NumpyEncoder)

    if write:
        with open("holog_obs_dict.json", "w") as outfile:
            outfile.write(encoded_obj)

    return HologObsDict(json.loads(encoded_obj))


def get_number_of_parameters(holog_obs_dict: HologObsDict) -> Tuple[int, int, int, int]:
    scan_list = []
    ant_list = []
    baseline_list = []

    for ddi in holog_obs_dict.keys():
        for mapping in holog_obs_dict[ddi].keys():
            scan_list.append(len(holog_obs_dict[ddi][mapping]["scans"]))
            ant_list.append(len(holog_obs_dict[ddi][mapping]["ant"].keys()))

            for ant in holog_obs_dict[ddi][mapping]["ant"].keys():
                baseline_list.append(len(holog_obs_dict[ddi][mapping]["ant"][ant]))

    n_ddi = len(holog_obs_dict.keys())
    n_scans = max(scan_list)
    n_ant = max(ant_list)
    n_baseline = max(baseline_list)

    return n_ddi, n_scans, n_ant, n_baseline


def model_memory_usage(
        ms_name: str,
        holog_obs_dict: HologObsDict = None
) -> int:
    """ Determine the approximate memory usage per core of a given measurement file.

        :param ms_name: Measurement file name
        :type ms_name: str

        :param holog_obs_dict: Holography observations dictionary.
        :type holog_obs_dict: HologObsDict, optional

        :return: Memory per core
        :rtype: int
    """

    # Get holog observations dictionary
    if holog_obs_dict is None:
        extract_pointing(
            ms_name=ms_name,
            point_name="temporary.pointing.zarr",
            parallel=False,
            overwrite=True,
        )

        holog_obs_dict = generate_holog_obs_dict(
            ms_name=ms_name,
            point_name="temporary.pointing.zarr",
            baseline_average_distance='all',
            baseline_average_nearest='all',
            parallel=False
        )

        shutil.rmtree("temporary.pointing.zarr")

    # Get number of each parameter
    n_ddi, n_scans, n_ant, n_baseline = get_number_of_parameters(holog_obs_dict)

    # Get model file
    if not pathlib.Path("model").exists():
        os.mkdir("model")

    toolviper.utils.data.download('heuristic_model', folder="model")

    with open("model/elastic.model", "rb") as model_file:
        model = pickle.load(model_file)

    memory_per_core = math.ceil(model.predict([[n_ddi, n_scans, n_ant, n_baseline]])[0])

    cores = multiprocessing.cpu_count()
    memory_available = round((psutil.virtual_memory().available / (1024 ** 2)))
    memory_limit = round((psutil.virtual_memory().total / (1024 ** 2)))

    table = Table(
        title="System Info",
        caption="Available memory: represents the system memory available without going into swap"
    )

    table.add_column("N-cores", justify="right", style="blue", no_wrap=True)
    table.add_column("Available memory (MB)", style="magenta")
    table.add_column("Total memory (MB)", style="cyan")
    table.add_column("Suggested memory per core (MB)", justify="right", style="green")

    table.add_row(str(cores), str(memory_available), str(memory_limit), str(memory_per_core))

    console = Console()
    console.print(table)

    # Make prediction of memory per core in MB
    return memory_per_core


def _convert_ant_name_to_id(ant_list, ant_names):
    """_summary_

  Args:
      ant_list (_type_): _description_
      ant_names (_type_): _description_

  Returns:
      _type_: _description_
  """

    return np.nonzero(np.in1d(ant_list, ant_names))[0]
