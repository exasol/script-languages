import gzip
import logging
import pathlib
import shutil
from datetime import datetime, timedelta

import luigi
from docker.models.containers import Container

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.log_config import log_config, WriteLogFilesToConsole
from exaslct_src.lib.test_runner.container_log_thread import ContainerLogThread
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.test_runner.is_database_ready_thread import IsDatabaseReadyThread


class WaitForTestDockerDatabase(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    environment_name = luigi.Parameter()
    test_container_info_dict = luigi.DictParameter(significant=False)
    database_info_dict = luigi.DictParameter(significant=False)
    db_startup_timeout_in_seconds = luigi.IntParameter(5 * 60, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = docker_config().get_client()
        self._low_level_client = docker_config().get_low_level_client()
        self._test_container_info = ContainerInfo.from_dict(self.test_container_info_dict)
        self._database_info = DatabaseInfo.from_dict(self.database_info_dict)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()
        self._low_level_client.close()

    def _prepare_outputs(self):
        self._database_ready_target = luigi.LocalTarget(
            "%s/info/environment/%s/database/%s/ready"
            % (build_config().output_directory,
               self.environment_name,
               self._database_info.container_info.container_name))
        if self._database_ready_target.exists():
            self._database_ready_target.remove()

    def output(self):
        return self._database_ready_target

    def run_task(self):
        test_container = self._client.containers.get(self._test_container_info.container_name)
        db_container_name = self._database_info.container_info.container_name
        db_container = self._client.containers.get(db_container_name)
        database_log_path = \
            pathlib.Path("%s/logs/environment/%s/database/%s/"
                         % (build_config().output_directory,
                            self.environment_name,
                            db_container_name))

        is_database_ready = \
            self.wait_for_database_startup(database_log_path, test_container, db_container)
        after_startup_db_log_file = database_log_path.joinpath("after_startup_db_log.tar.gz")
        self.save_db_log_files_as_gzip_tar(after_startup_db_log_file, db_container)
        if not is_database_ready:
            raise Exception("Startup of database in container %s failed" % db_container.name)
        else:
            self.write_output()

    def wait_for_database_startup(self, database_log_path,
                                  test_container: Container,
                                  db_container: Container):
        container_log_thread, is_database_ready_thread = \
            self.start_wait_threads(database_log_path, db_container, test_container)
        is_database_ready = \
            self.wait_for_threads(container_log_thread, is_database_ready_thread)
        self.join_threads(container_log_thread, is_database_ready_thread)
        if not is_database_ready:
            if log_config().write_log_files_to_console == WriteLogFilesToConsole.only_error:
                self.logger.error("Task %s: Database startup failed %s failed\nStartup Log:\n%s",
                                  self.task_id,
                                  db_container.name,
                                  "\n".join(container_log_thread.complete_log))
        return is_database_ready

    def start_wait_threads(self, database_log_path, db_container, test_container):
        startup_log_file = self.prepare_startup_log_file(database_log_path)
        container_log_thread = ContainerLogThread(db_container,
                                                  self.__repr__(),
                                                  self.logger,
                                                  startup_log_file,
                                                  "Database Startup %s" % db_container.name)
        container_log_thread.start()
        is_database_ready_thread = IsDatabaseReadyThread(self.__repr__(),
                                                         self._database_info, test_container,
                                                         self.db_startup_timeout_in_seconds)
        is_database_ready_thread.start()
        return container_log_thread, is_database_ready_thread

    def prepare_startup_log_file(self, database_log_path):
        if database_log_path.exists():
            shutil.rmtree(database_log_path)
        database_log_path.mkdir(parents=True)
        startup_log_file = database_log_path.joinpath("startup.log")
        return startup_log_file

    def join_threads(self, container_log_thread: ContainerLogThread,
                     is_database_ready_thread: IsDatabaseReadyThread):
        container_log_thread.stop()
        is_database_ready_thread.stop()
        container_log_thread.join()
        is_database_ready_thread.join()

    def wait_for_threads(self, container_log_thread: ContainerLogThread,
                         is_database_ready_thread: IsDatabaseReadyThread):
        is_database_ready = False
        start_time = datetime.now()
        time_delta = timedelta(minutes=5)
        while (True):
            if container_log_thread.error_message != None:
                is_database_ready = False
                break
            if is_database_ready_thread.finish:
                is_database_ready = True
                break

            if datetime.now() - start_time > time_delta:
                self.logger.warning(
                    "Task %s: Database startup didn't finished after %s, here some debug information %s",
                    self.__repr__(), datetime.now() - start_time > time_delta,
                    f"""
                    ========== Container-Log: ============ 
                    {container_log_thread.complete_log}
                    ========== IsDatabaseReadyThread output db connection: ============
                    {is_database_ready_thread.output_db_connection}
                    ========== IsDatabaseReadyThread output db connection: ============
                    {is_database_ready_thread.output_bucketfs_connection}
                    """)
                time_delta=time_delta+timedelta(minutes=2)
        return is_database_ready

    def save_db_log_files_as_gzip_tar(self, path: pathlib.Path, database_container: Container):
        stream, stat = database_container.get_archive("/exa/logs/db")
        with gzip.open(path, "wb") as file:
            for chunk in stream:
                file.write(chunk)

    def write_output(self):
        with self.output().open("w") as file:
            file.write("READY")
