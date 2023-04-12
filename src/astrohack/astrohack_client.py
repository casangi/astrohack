import warnings, time, os, psutil, multiprocessing, re
import dask
import copy
import os
import logging
import astrohack
from astrohack._utils._parm_utils._check_logger_parms import _check_logger_parms, _check_worker_logger_parms
from astrohack._utils._logger._astrohack_logger import  _setup_astrohack_logger, _get_astrohack_logger
from astrohack._utils._dask_plugins._astrohack_worker import _astrohack_worker

def astrohack_local_client(cores=None, memory_limit=None,astrohack_autorestrictor=False,dask_local_dir=None,astrohack_local_dir=None,wait_for_workers=True, log_parms={}, worker_log_parms={}):
    '''
    astrohack_local_dir setting is only useful for testing since this function creates a local cluster. astrohack_slurm_cluster_client should be used for a multinode cluster.

    https://github.com/dask/dask/issues/5577
    log_parms['log_to_term'] = True/False
    log_parms['log_file'] = True/False
    log_parms['log_level'] =
    '''

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


def astrohack_slurm_cluster_client(workers_per_node, cores_per_node, memory_per_node, number_of_nodes, queue, interface, python_env_dir, dask_local_dir, dask_log_dir, exclude_nodes='nmpost090', dashboard_port=9000, astrohack_local_dir=None,astrohack_autorestrictor=False,wait_for_workers=True, log_parms={}, worker_log_parms={}):

    '''
    local_cache setting is only useful for testing since this function creates a local cluster. astrohack_slurm_cluster_client should be used for a multinode cluster.

    https://github.com/dask/dask/issues/5577
    log_parms['log_to_term'] = True/False
    log_parms['log_file'] = True/False
    log_parms['log_level'] =
    
    interface eth0, ib0
    python "/mnt/condor/jsteeb/astrohack_py/bin/python"
    dask_local_dir "/mnt/condor/jsteeb"
    dask_log_dir "/.lustre/aoc/projects/ngvla/astrohack/ngvla_sim",
    '''
    
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, config, performance_report
    
    _log_parms = copy.deepcopy(log_parms)
    _worker_log_parms = copy.deepcopy(worker_log_parms)
    
    assert(_check_logger_parms(_log_parms)), "######### ERROR: initialize_processing log_parms checking failed."
    assert(_check_worker_logger_parms(_worker_log_parms)), "######### ERROR: initialize_processing log_parms checking failed."
    
    if astrohack_local_dir:
        os.environ['ASTROHACK_LOCAL_DIR'] = astrohack_local_dir
        local_cache = True
    else:
        local_cache = False
    
    #astrohack logger for code that is not part of the Dask graph. The worker logger is setup in the _astrohack_worker plugin.
    from astrohack._utils._astrohack_logger import _setup_astrohack_logger
    _setup_astrohack_logger(**_log_parms)
    logger = _get_astrohack_logger()

    _set_up_dask(dask_local_dir)
 
    astrohack_path = astrohack.__path__.__dict__["_path"][0]
    if local_cache or astrohack_autorestrictor:
        dask.config.set({"distributed.scheduler.preload": os.path.join(astrohack_path,'_utils/_astrohack_scheduler.py')})
        dask.config.set({"distributed.scheduler.preload-argv": ["--local_cache",local_cache,"--autorestrictor",astrohack_autorestrictor]})
    
    
    ''' This method of assigning a worker plugin does not seem to work when using dask_jobqueue. Consequently using client.register_worker_plugin so that the method of assigning a worker plugin is the same for astrohack_local_client and astrohack_slurm_cluster_client.
    if local_cache or _worker_log_parms:
        dask.config.set({"distributed.worker.preload": os.path.join(astrohack_path,'_utils/_astrohack_worker.py')})
        dask.config.set({"distributed.worker.preload-argv": ["--local_cache",local_cache,"--log_to_term",_worker_log_parms['log_to_term'],"--log_to_file",_worker_log_parms['log_to_file'],"--log_file",_worker_log_parms['log_file'],"--log_level",_worker_log_parms['log_level']]})
    '''
    
    cluster = SLURMCluster(
        processes=workers_per_node,
        cores=cores_per_node,
        interface=interface,
        memory=memory_per_node,
        walltime="24:00:00",
        queue=queue,
        name="astrohack",
        python=python_env_dir, #"/mnt/condor/jsteeb/astrohack_py/bin/python", #"/.lustre/aoc/projects/ngvla/astrohack/astrohack_py_env/bin/python",
        local_directory=dask_local_dir, #"/mnt/condor/jsteeb",
        log_directory=dask_log_dir,
        job_extra_directives=["--exclude="+exclude_nodes],
        #job_extra_directives=["--exclude=nmpost087,nmpost089,nmpost088"],
        scheduler_options={"dashboard_address": ":"+str(dashboard_port)}
    )  # interface='ib0'
    
    client = Client(cluster)


    cluster.scale(workers_per_node*number_of_nodes)

    '''
    When constructing a graph that has local cache enabled all workers need to be up and running.
    '''
    if local_cache or wait_for_workers:
        client.wait_for_workers(n_workers=workers_per_node*number_of_nodes)

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
 


