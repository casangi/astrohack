import os
import distributed

from astrohack._utils._dask_graph_tools import _construct_graph, _dask_compute

from astrohack.dio import open_holog

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._diagnostics import _calibration_plot_chunk

def plot_holog_diagnostics(
    file, 
    delta=0.01,
    out_folder="", 
    ant_id="", 
    ddi="", 
    map_id="", 
    data_type='amplitude',
    save_plots=False,
    display=True,
    width=1250, 
    height=1200,
    parallel=False
):
  """ Plot diagnostic calibration plots from the holography data file. 

  :param file: AstroHack holography file, ie. *.holog.zarr
  :type file: str

  :param delta: Defines a fraction of cell_size around which to look fro peaks., defaults to 0.01
  :type delta: float, optional

  :param ant_id: antenna ID to use in subselection, defaults to ""
  :type ant_id: str, optional

  :param ddi: data description ID to use in subselection, defaults to ""
  :type ddi: str, optional

  :param map_id: map ID to use in subselection. This relates to which antenna are in the mapping vs. scanning configuration,  defaults to ""
  :type map_id: str, optional

  :param data_type: Whether the plots should investigate amplitude/phase or real/imaginary. Options are 'amplitude' or 'real', defaults to 'amplitude'
  :type data_type: str, optional

  :param save_plots: Save plots to disk, defaults to False
  :type save_plots: bool, optional

  :param display: Display plots inline or suppress, defaults to True
  :type display: bool, optional

  :param width: figure width in pixels, defaults to 1250
  :type width: int, optional

  :param height: figure height in pixels, defaults to 1200
  :type height: int, optional

  :param parallel: Run inparallel, defaults to False
  :type parallel: bool, optional
  """

  # This is the default address used by Dask. Note that in the client check below, if the user has multiple clients running
  # a new client may still be spawned but only once. If run again in a notebook session the local_client check will catch it.
  # It will also be caught if the user spawns their own instance in the notebook.
  DEFAULT_DASK_ADDRESS="127.0.0.1:8786" 

  logger = _get_astrohack_logger()

  if parallel:
    if not distributed.client._get_global_client():
      try:
        Client(DEFAULT_DASK_ADDRESS, timeout=2)

      except Exception:
        from astrohack.astrohack_client import astrohack_local_client

        logger.info("local client not found, starting ...")

        log_parms = {'log_level':'DEBUG'}
        client = astrohack_local_client(cores=2, memory_limit='8GB', log_parms=log_parms)
        logger.info(client.dashboard_link)

  hack_mds = open_holog(file)

  if save_plots:
    os.makedirs("{out_folder}/".format(out_folder=out_folder), exist_ok=True)

    
  # Default but ant | ddi | map take precendence
  key_list = ["ant_", "ddi_", "map_"]
    
  if ant_id or ddi or map_id: 
    key_list = []
        
    key_list.append("ant_") if not ant_id else key_list.append(ant_id)
    key_list.append("ddi_") if not ddi else key_list.append(ddi)
    key_list.append("map_") if not map_id else key_list.append(map_id)
    
  param_dict = {
    'data': None,
    'delta': delta,
    'type': data_type,
    'save': save_plots,
    'display': display,
    'width': width,
    'height': height,
    'out_folder': out_folder
  }
    
  _dask_compute(
    data_dict=hack_mds, 
    function=_calibration_plot_chunk, 
    param_dict=param_dict, 
    key_list=key_list, 
    parallel=parallel
  )
  
  

