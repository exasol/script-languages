import gzip
import logging
import pathlib
import shutil
import time
from datetime import datetime, timedelta

import luigi
from docker.models.containers import Container

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.test_runner.container_log_thread import ContainerLogThread
from exaslct_src.lib.test_runner.is_database_ready_thread import IsDatabaseReadyThread


class WaitForTestExternalDatabase(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    environment_name = luigi.Parameter()
    test_container_info_dict = luigi.DictParameter(significant=False)
    database_info_dict = luigi.DictParameter(significant=False)
    db_startup_timeout_in_seconds = luigi.IntParameter(1*60, significant=False)
    attempt = luigi.IntParameter(1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = docker_client_config().get_client()
        self._low_level_client = docker_client_config().get_low_level_client()
        self._test_container_info = ContainerInfo.from_dict(self.test_container_info_dict)
        self._database_info = DatabaseInfo.from_dict(self.database_info_dict)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()
        self._low_level_client.close()

    def _prepare_outputs(self):
        if self._database_info.container_info is None:
            database_name = "external-db"
        else:
            raise Exception("container_info not set for database_info %s" % self.database_info_dict)
        self._database_ready_target = luigi.LocalTarget(
            "%s/info/environment/%s/database/%s/%s/ready"
            % (build_config().output_directory,
               self.environment_name,
               database_name,
               self.attempt))
        if self._database_ready_target.exists():
            self._database_ready_target.remove()

    def output(self):
        return self._database_ready_target

    def run_task(self):
        test_container = self._client.containers.get(self._test_container_info.container_name)
        is_database_ready = self.wait_for_database_startup(test_container)
        self.write_output(is_database_ready)

    def wait_for_database_startup(self, test_container: Container):
        is_database_ready_thread = self.start_wait_threads(test_container)
        is_database_ready = self.wait_for_threads(is_database_ready_thread)
        self.join_threads(is_database_ready_thread)
        return is_database_ready

    def start_wait_threads(self, test_container):
        is_database_ready_thread = IsDatabaseReadyThread(self.__repr__(),
                                                         self._database_info, test_container)
        is_database_ready_thread.start()
        return is_database_ready_thread

    def join_threads(self, is_database_ready_thread: IsDatabaseReadyThread):
        is_database_ready_thread.stop()
        is_database_ready_thread.join()

    def wait_for_threads(self, is_database_ready_thread: IsDatabaseReadyThread):
        is_database_ready = False
        reason = None
        start_time = datetime.now()
        while (True):
            if is_database_ready_thread.finish:
                is_database_ready = True
                break
            if self.timeout_occured(start_time):
                reason = f"timeout after after {self.db_startup_timeout_in_seconds} seconds"
                is_database_ready = False
                break
            time.sleep(1)
        if not is_database_ready:
            self.log_database_not_ready(is_database_ready_thread, reason)
        is_database_ready_thread.stop()
        return is_database_ready

    def log_database_not_ready(self, is_database_ready_thread, reason):
        log_information = f"""
========== IsDatabaseReadyThread output db connection: ============
{is_database_ready_thread.output_db_connection}
========== IsDatabaseReadyThread output bucketfs connection: ============
{is_database_ready_thread.output_bucketfs_connection}
"""
        self.logger.warning(
            'Task %s: Database startup failed for following reason "%s", here some debug information \n%s',
            self.__repr__(), reason, log_information)

    def timeout_occured(self, start_time):
        timeout = timedelta(seconds=self.db_startup_timeout_in_seconds)
        return datetime.now() - start_time > timeout

    def write_output(self, is_database_ready: bool):
        with self.output().open("w") as file:
            file.write(str(is_database_ready))
