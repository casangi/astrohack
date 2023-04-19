import warnings, time, os, psutil, multiprocessing, re
import dask
import copy
import os
import logging
import astrohack
from astrohack._utils._parm_utils._check_logger_parms import _check_logger_parms, _check_worker_logger_parms
from astrohack._utils._logger._astrohack_logger import  _setup_astrohack_logger, _get_astrohack_logger
from astrohack._utils._dask_plugins._astrohack_worker import _astrohack_worker

def astrohack_local_client(cores=None, memory_limit=None, dask_local_dir=None, log_parms={}, worker_log_parms={}):
    """Setup dask cluster and astrohack logger.

    :param cores: Number of cores in Dask cluster.
    :type cores: int

    :param memory_limit: Amount of memory per core. It is suggested to use '8GB'.
    :type memory_limit: str

    :param dask_local_dir: Where Dask should store temporary files, defaults to None. If None Dask will use ./dask-worker-space.
    :type dask_local_dir: str
   
    :param log_parms: The logger for the main process (code that does not run in parallel),
    :type log_parms: dict, optional
 
    :param log_parms['log_to_term']: Prints logging statements to the terminal, default to True.
    :type log_parms['log_to_term']: bool, optional
   
    :param log_parms['log_level']: Log level options are: 'CRITICAL', 'ERROR', 'WARNING', 'INFO', and 'DEBUG'. With defaults of 'INFO'.
    :type log_parms['log_level']: bool, optional
   
    :param log_parms['log_to_file']: Write log to file, defaults to False.
    :type log_parms['log_to_file']: bool, optional
   
    :param log_parms['log_file']: If log_parms['log_to_file'] is True the log will be written to a file with the name log_parms['log_file'],
    :type log_parms['log_file']: bool, optional
   
    :param worker_log_parms: has the same keys as log_parms. However the defaults are {'log_to_term':False,'log_level':'INFO','log_to_file':False,'log_file':None}.
    :type dict
    """
    
    #Secret parameters user do not need to know about.
    astrohack_autorestrictor=False
    wait_for_workers=True
    
    astrohack_local_dir=None #Not needed for a local cluster, but useful for testing.

    _log_parms = copy.deepcopy(log_parms)
    _worker_log_parms = copy.deepcopy(worker_log_parms)
    
    assert(_check_logger_parms(_log_parms)), "######### ERROR: initialize_processing log_parms checking failed."
    assert(_check_worker_logger_parms(_worker_log_parms)), "######### ERROR: initialize_processing log_parms checking failed."
    
    if astrohack_local_dir:
        os.environ['ASTROHACK_LOCAL_DIR'] = astrohack_local_dir
        local_cache = True
    else:
        local_cache = False
    
    _setup_astrohack_logger(**_log_parms)
    logger = _get_astrohack_logger()
    
    _set_up_dask(dask_local_dir)
    
    #astrohack_path = astrohack.__path__.__dict__["_path"][0]
    astrohack_path = astrohack.__path__[0]
    
    if local_cache or astrohack_autorestrictor:
        dask.config.set({"distributed.scheduler.preload": os.path.join(astrohack_path,'_utils/_astrohack_scheduler.py')})
        dask.config.set({"distributed.scheduler.preload-argv": ["--local_cache",local_cache,"--autorestrictor",astrohack_autorestrictor]})
    
    
    ''' This method of assigning a worker plugin does not seem to work when using dask_jobqueue. Consequently using client.register_worker_plugin so that the method of assigning a worker plugin is the same for astrohack_local_client and astrohack_slurm_cluster_client.
    if local_cache or _worker_log_parms:
        dask.config.set({"distributed.worker.preload": os.path.join(astrohack_path,'_utils/_astrohack_worker.py')})
        dask.config.set({"distributed.worker.preload-argv": ["--local_cache",local_cache,"--log_to_term",_worker_log_parms['log_to_term'],"--log_to_file",_worker_log_parms['log_to_file'],"--log_file",_worker_log_parms['log_file'],"--log_level",_worker_log_parms['log_level']]})
    '''
    # setup dask.distributed based multiprocessing environment
    if cores is None: cores = multiprocessing.cpu_count()
    if memory_limit is None: memory_limit = str(round(((psutil.virtual_memory().available / (1024 ** 2))) / cores)) + 'MB'
    
    #print('cores',cores,memory_limit)
    
    cluster = dask.distributed.LocalCluster(n_workers=cores, threads_per_worker=1, processes=True, memory_limit=memory_limit,silence_logs=logging.ERROR) #, silence_logs=logging.ERROR #,resources={'GPU': 2}
    client = dask.distributed.Client(cluster)
    client.get_versions(check=True)
    
    #print(client)

    '''
    When constructing a graph that has local cache enabled all workers need to be up and running.
    '''
    if local_cache or wait_for_workers:
        client.wait_for_workers(n_workers=cores)

    if local_cache or _worker_log_parms:
        plugin = _astrohack_worker(local_cache,_worker_log_parms)
        client.register_worker_plugin(plugin,name='astrohack_worker')
        
    
    logger.info('Created client ' + str(client))
    
    return client
    
def _set_up_dask(local_directory):
    if local_directory: dask.config.set({"temporary_directory": local_directory})
    dask.config.set({"distributed.scheduler.allowed-failures": 10})
    dask.config.set({"distributed.scheduler.work-stealing": True})
    dask.config.set({"distributed.scheduler.unknown-task-duration": '99m'})
    dask.config.set({"distributed.worker.memory.pause": False})
    dask.config.set({"distributed.worker.memory.terminate": False})
    #dask.config.set({"distributed.worker.memory.recent-to-old-time": '999s'})
    dask.config.set({"distributed.comm.timeouts.connect": '3600s'})
    dask.config.set({"distributed.comm.timeouts.tcp": '3600s'})
    dask.config.set({"distributed.nanny.environ.OMP_NUM_THREADS": 1})
    dask.config.set({"distributed.nanny.environ.MKL_NUM_THREADS": 1})
    #https://docs.dask.org/en/stable/how-to/customize-initialization.html
 


