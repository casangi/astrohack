import sys

import click
import skriba.logger as logger

from distributed.diagnostics.plugin import WorkerPlugin


class DaskWorker(WorkerPlugin):
    def __init__(self, local_cache, log_params):
        self.logger = None
        self.worker = None

        self.local_cache = local_cache
        self.log_to_term = log_params['log_to_term']
        self.log_to_file = log_params['log_to_file']
        self.log_file = log_params['log_file']
        self.log_level = log_params['log_level']

    def get_logger(self):
        return self.logger

    def setup(self, worker):
        """
        Run when the plugin is attached to a worker. This happens when the plugin is registered
        and attached to existing workers, or when a worker is created after the plugin has been
        registered.
        """

        self.logger = logger.setup_worker_logger(
            logger_name="dask-worker",
            log_to_term=self.log_to_term,
            log_to_file=self.log_to_file,
            log_file=self.log_file,
            log_level=self.log_level,
            worker_id=worker.id
        )

        self.logger.debug('Logger created on worker ' + str(worker.id) + ',*,' + str(worker.address))

        # Documentation https://distributed.dask.org/en/stable/worker.html#distributed.worker.Worker
        self.worker = worker

        if self.local_cache:
            ip = worker.address[worker.address.rfind('/') + 1:worker.address.rfind(':')]

            self.logger.debug(str(worker.id) + ',*,' + ip)

            worker.state.available_resources = {**worker.state.available_resources, **{ip: 1}}


# https://github.com/dask/distributed/issues/4169
@click.command()
@click.option("--local_cache", default=False)
@click.option("--log_to_term", default=True)
@click.option("--log_to_file", default=False)
@click.option("--log_file", default='dask-worker-')
@click.option("--log_level", default='INFO')
async def dask_setup(worker, local_cache, log_to_term, log_to_file, log_file, log_level):
    log_params = {
        'log_to_term': log_to_term,
        'log_to_file': log_to_file,
        'log_file': log_file,
        'log_level': log_level
    }

    plugin = DaskWorker(local_cache, log_params)

    if sys.version_info.major == 3:
        if sys.version_info.minor > 8:
            await worker.client.register_plugin(plugin, name='worker_logger')

        else:
            await worker.client.register_worker_plugin(plugin, name='worker_logger')

    else:
        logger.warning("Python version may not be supported.")
