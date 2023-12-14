import psutil
import multiprocessing
import pathlib
import dask
import copy
import os
import yaml
import logging
import astrohack

import skriba.logger
import auror.parameter

from astrohack._utils._dask_plugins._astrohack_worker import AstrohackWorker


@auror.parameter.validate(
    logger=skriba.logger.get_logger(logger_name="astrohack")
)
def local_client(
        cores: int = None,
        memory_limit: str = None,
        dask_local_dir: str = None,
        log_params: dict = None,
        worker_log_params: dict = None
) -> dask.distributed.Client:
    """ Setup dask cluster and astrohack logger.

    :param cores: Number of cores in Dask cluster, defaults to None
    :type cores: int, optional

    :param memory_limit: Amount of memory per core. It is suggested to use '8GB', defaults to None
    :type memory_limit: str, optional
    
    :param dask_local_dir: Where Dask should store temporary files, defaults to None. If None Dask will use \
    `./dask-worker-space`, defaults to None
    :type dask_local_dir: str, optional

    :param log_params: The logger for the main process (code that does not run in parallel), defaults to {}
    :type log_params: dict, optional

    :param log_params['log_to_term']: Prints logging statements to the terminal, default to True.
    :type log_params['log_to_term']: bool, optional
   
    :param log_params['log_level']: Log level options are: 'CRITICAL', 'ERROR', 'WARNING', 'INFO', and 'DEBUG'. \
    With defaults of 'INFO'.
    :type log_params['log_level']: bool, optional
   
    :param log_params['log_to_file']: Write log to file, defaults to False.
    :type log_params['log_to_file']: bool, optional
   
    :param log_params['log_file']: If log_params['log_to_file'] is True the log will be written to a file with the name \
    log_params['log_file'],
    :type log_params['log_file']: bool, optional

    :param worker_log_params: worker_log_params: Keys as same as log_params, default values given in `Additional \
    Information`_.
    :type worker_log_params: dict, optional

    :return: Dask Distributed Client
    :rtype: distributed.Client
    
    
    .. _Additional Information:

    **Additional Information**
    
    ``worker_log_params`` default values are set internally when there is not user input. The default values are given\
     below.
    
    .. parsed-literal::
        worker_log_params =
            {
                'log_to_term':False,
                'log_level':'INFO',
                'log_to_file':False,
                'log_file':None
            }

    **Example Usage**
    
    .. parsed-literal::
        from astrohack.client import astrohack_local_client

        client = astrohack_local_client(
            cores=2, 
            memory_limit='8GB', 
            log_params={
                'log_level':'DEBUG'
            }
        )

    """

    if log_params is None:
        log_params = {
            'log_to_term': True,
            'log_level': 'INFO',
            'log_to_file': False,
            'log_file': None
        }

    if worker_log_params is None:
        worker_log_params = {
            'log_to_term': True,
            'log_level': 'INFO',
            'log_to_file': False,
            'log_file': None
        }

    # Secret parameters user do not need to know about.
    autorestrictor = False
    wait_for_workers = True

    # Not needed for a local cluster, but useful for testing.
    client_local_dir = None

    _log_params = copy.deepcopy(log_params)
    _worker_log_params = copy.deepcopy(worker_log_params)

    if client_local_dir:
        os.environ['CLIENT_LOCAL_DIR'] = client_local_dir
        local_cache = True

    else:
        local_cache = False

    skriba.logger.setup_logger(**_log_params)
    logger = skriba.logger.get_logger(logger_name="astrohack")

    if dask_local_dir is None:
        logger.warning("It is recommended that the local cache directory be set using the `local_dir` parameter.")

    _set_up_dask(dask_local_dir)

    # Need to generalize
    astrohack_path = astrohack.__path__[0]

    if local_cache or autorestrictor:
        # Also need to generalize
        dask.config.set({
            "distributed.scheduler.preload": os.path.join(astrohack_path, '_utils/_astrohack_scheduler.py')
        })

        dask.config.set({
            "distributed.scheduler.preload-argv": [
                "--local_cache", local_cache,
                "--autorestrictor", autorestrictor
            ]
        })

    ''' This method of assigning a worker plugin does not seem to work when using dask_jobqueue. Consequently using \
    client.register_worker_plugin so that the method of assigning a worker plugin is the same for astrohack_local_client\
     and astrohack_slurm_cluster_client.
    if local_cache or _worker_log_params:
        dask.config.set({"distributed.worker.preload": os.path.join(astrohack_path,'_utils/_astrohack_worker.py')})
        dask.config.set({"distributed.worker.preload-argv": ["--local_cache",local_cache,"--log_to_term",\
        _worker_log_params['log_to_term'],"--log_to_file",_worker_log_params['log_to_file'],"--log_file",\
        _worker_log_params['log_file'],"--log_level",_worker_log_params['log_level']]})
    '''
    # setup dask.distributed based multiprocessing environment
    if cores is None:
        cores = multiprocessing.cpu_count()

    if memory_limit is None:
        memory_limit = str(round((psutil.virtual_memory().available / (1024 ** 2)) / cores)) + 'MB'

    cluster = dask.distributed.LocalCluster(
        n_workers=cores,
        threads_per_worker=1,
        processes=True,
        memory_limit=memory_limit,
        silence_logs=logging.ERROR  # , silence_logs=logging.ERROR #,resources={ 'GPU': 2}
    )

    client = dask.distributed.Client(cluster)
    client.get_versions(check=True)

    '''
    When constructing a graph that has local cache enabled all workers need to be up and running.
    '''
    if local_cache or wait_for_workers:
        client.wait_for_workers(n_workers=cores)

    if local_cache or _worker_log_params:
        plugin = AstrohackWorker(local_cache, _worker_log_params)
        client.register_worker_plugin(plugin, name='worker_logger')

    logger.info('Created client ' + str(client))

    return client


def _set_up_dask(local_directory: str) -> None:
    if local_directory:
        dask.config.set({"temporary_directory": local_directory})

    config_path = pathlib.Path(__file__).joinpath("config")
    if config_path.exists():
        config_file = str(config_path.joinpath("dask.config.yaml"))
        with open(config_file) as config:
            dask_settings = yaml.safe_load(config)
            dask.config.update_defaults(dask_settings)
    '''
    dask.config.set({"distributed.scheduler.allowed-failures": 10})
    dask.config.set({"distributed.scheduler.work-stealing": True})
    dask.config.set({"distributed.scheduler.unknown-task-duration": '99m'})
    dask.config.set({"distributed.worker.memory.pause": False})
    dask.config.set({"distributed.worker.memory.terminate": False})
    # dask.config.set({"distributed.worker.memory.recent-to-old-time": '999s'})
    dask.config.set({"distributed.comm.timeouts.connect": '3600s'})
    dask.config.set({"distributed.comm.timeouts.tcp": '3600s'})
    dask.config.set({"distributed.nanny.environ.OMP_NUM_THREADS": 1})
    dask.config.set({"distributed.nanny.environ.MKL_NUM_THREADS": 1})

    # https://docs.dask.org/en/stable/how-to/customize-initialization.html
    '''
