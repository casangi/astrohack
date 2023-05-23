import os
import distributed

from astrohack._utils._tools import _dask_compute
from astrohack._utils._tools import _construct_graph

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
    width=1000, 
    height=450,
    parallel=False
):
  """_summary_

  :param file: _description_
  :type file: _type_
  :param delta: _description_, defaults to 0.01
  :type delta: float, optional
  :param ant_id: _description_, defaults to ""
  :type ant_id: str, optional
  :param ddi: _description_, defaults to ""
  :type ddi: str, optional
  :param map_id: _description_, defaults to ""
  :type map_id: str, optional
  :param data_type: _description_, defaults to 'amplitude'
  :type data_type: str, optional
  :param save_plots: _description_, defaults to False
  :type save_plots: bool, optional
  :param display: _description_, defaults to True
  :type display: bool, optional
  :param width: _description_, defaults to 1000
  :type width: int, optional
  :param height: _description_, defaults to 450
  :type height: int, optional
  :param parallel: _description_, defaults to False
  :type parallel: bool, optional
  """

  logger = _get_astrohack_logger()

  if parallel:
    if not distributed.client._get_global_client():
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
  
  

