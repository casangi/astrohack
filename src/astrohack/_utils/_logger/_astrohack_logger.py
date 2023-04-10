 #   Copyright 2019 AUI, Inc. Washington DC, USA
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


hack_logger_name = 'astrohack'
import sys
import logging
from datetime import datetime
#formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
#formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
from dask.distributed import WorkerPlugin
import dask


class astrohack_formatter(logging.Formatter):

    reset = "\x1b[0m"
    DEBUG = "\x1b[32;20m"   #green
    INFO = "\x1b[33;34m"    #blue
    WARNING = "\x1b[33;33m" #yellow
    ERROR = "\x1b[32;31m"   #red
    CRITICAL = "\x1b[31;1m" #bold red

    start_msg = "%(asctime)s - "
    middle_msg = "%(levelname)-8s"
    end_msg = " - %(name)s - (%(filename)s:%(lineno)d) - %(message)s"
    
    FORMATS = {
        logging.DEBUG: start_msg + DEBUG + middle_msg  + reset + end_msg ,
        logging.INFO:  start_msg + INFO + middle_msg  + reset + end_msg ,
        logging.WARNING:  start_msg + WARNING + middle_msg  + reset + end_msg ,
        logging.ERROR:  start_msg + ERROR + middle_msg  + reset + end_msg ,
        logging.CRITICAL:  start_msg + CRITICAL + middle_msg  + reset + end_msg ,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def _get_astrohack_logger(name = hack_logger_name):
    '''
    Will first try to get worker logger. If this fails graph construction logger is returned.
    '''
    from dask.distributed import get_worker
    try:
        worker = get_worker()
    except:
        return logging.getLogger(name)
    
    try:
        logger = worker.plugins['astrohack_worker'].get_logger()
        return logger
    except:
        return logging.getLogger()

def _setup_astrohack_logger(log_to_term=False,log_to_file=True,log_file='astrohack_', log_level='INFO', name=hack_logger_name):
    """To setup as many loggers as you want"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(log_level))
    
    logger.handlers.clear()
    
    if log_to_term:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(astrohack_formatter())
        logger.addHandler(handler)
    
    if log_to_file:
        log_file = log_file+datetime.today().strftime('%Y%m%d_%H%M%S')+'.log'
        handler = logging.FileHandler(log_file)
        handler.setFormatter(astrohack_formatter())
        logger.addHandler(handler)
        
    return logger
    
def _get_astrohack_worker_logger_name(name=hack_logger_name):
    from dask.distributed import get_worker
    worker_log_name = name + '_' + str(get_worker().id)
    return worker_log_name

'''
class _astrohack_worker_logger_plugin(WorkerPlugin):
    def __init__(self,log_parms):
        self.log_to_term=log_parms['log_to_term']
        self.log_to_file=log_parms['log_to_file']
        self.log_file=log_parms['log_file']
        self.level=log_parms['log_level']
        self.logger = None
        print(self.log_to_term,self.log_to_file,self.log_file,self.log_level)
        
    def get_logger(self):
        return self.logger
        
    def setup(self, worker: dask.distributed.Worker):
        "Run when the plugin is attached to a worker. This happens when the plugin is registered and attached to existing workers, or when a worker is created after the plugin has been registered."
        self.logger = _setup_astrohack_worker_logger(self.log_to_term,self.log_to_file,self.log_file,self.level)
'''

def _setup_astrohack_worker_logger(log_to_term,log_to_file,log_file, log_level, worker_id):
    from dask.distributed import get_worker
    #parallel_logger_name = _get_astrohack_worker_logger_name()
    parallel_logger_name = hack_logger_name + '_' + str(worker_id)
    
    logger = logging.getLogger(parallel_logger_name)
    logger.setLevel(logging.getLevelName(log_level))
    
    if log_to_term:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(astrohack_formatter())
        logger.addHandler(handler)
    
    if log_to_file:
        log_file = log_file + '_' + str(worker_id) + '_' + datetime.today().strftime('%Y%m%d_%H%M%S') + '.log'
        handler = logging.FileHandler(log_file)
        handler.setFormatter(astrohack_formatter())
        logger.addHandler(handler)
    return logger
