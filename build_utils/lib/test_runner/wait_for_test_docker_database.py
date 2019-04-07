import gzip
import logging
import pathlib
import shutil
import time
from threading import Thread

import docker
import luigi
from docker.models.containers import Container

from build_utils.lib.build_config import build_config
from build_utils.lib.data.container_info import ContainerInfo
from build_utils.lib.data.database_info import DatabaseInfo
from build_utils.lib.docker_config import docker_config
from build_utils.lib.log_config import log_config, WriteLogFilesToConsole
from build_utils.lib.test_runner.container_log_thread import ContainerLogThread
from build_utils.stoppable_task import StoppableTask


class IsDatabaseReadyThread(Thread):
    def __init__(self, database_info: DatabaseInfo, test_container: Container, timeout_in_seconds: int):
        super().__init__()
        self._database_info = database_info
        self.test_container = test_container
        self.timeout_in_seconds = timeout_in_seconds
        self.finish = False
        self.is_ready = False

    def stop(self):
        self.finish = True

    def run(self):
        start_time = time.time()
        timeout_over = lambda current_time: current_time - start_time > self.timeout_in_seconds
        username = "sys"
        password = "exasol"
        connection_options = f"""-c '{self._database_info.host}:{self._database_info.db_port}' -u '{username}' -p '{password}'"""
        cmd = f"""$EXAPLUS {connection_options}  -sql 'select 1;'"""
        bash_cmd = f"""bash -c "{cmd}" """
        while not timeout_over(time.time()) and not self.finish:
            (exit_code, output) = \
                self.test_container.exec_run(cmd=bash_cmd)
            if exit_code == 0:
                self.finish = True
                self.is_ready = True
            time.sleep(1)


class WaitForTestDockerDatabase(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    test_container_info_dict = luigi.DictParameter(significant=False)
    database_info_dict = luigi.DictParameter(significant=False)
    db_startup_timeout_in_seconds = luigi.IntParameter(5 * 60, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._low_level_client = docker.APIClient(base_url=self._docker_config.base_url)
        self._test_container_info = ContainerInfo.from_dict(self.test_container_info_dict)
        self._database_info = DatabaseInfo.from_dict(self.database_info_dict)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()
        self._low_level_client.close()

    def _prepare_outputs(self):
        self._database_ready_target = luigi.LocalTarget(
            "%s/test-runner/db-test/database/%s/ready"
            % (self._build_config.output_directory,
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
            pathlib.Path("%s/logs/test-runner/db-test/database/%s/"
                         % (self._build_config.output_directory,
                            db_container_name))

        is_database_ready = \
            self.wait_for_database_startup(
                database_log_path,
                test_container,
                db_container
            )
        after_startup_db_log_file = database_log_path.joinpath("after_startup_db_log.tar.gz")
        self.save_db_log_files_as_gzip_tar(after_startup_db_log_file, db_container)
        if not is_database_ready:
            raise Exception("Startup of database in container %s failed" % db_container.name)
        else:
            self.write_output()

    def wait_for_database_startup(self, database_log_path,
                                  test_container: Container,
                                  db_container: Container):
        if database_log_path.exists():
            shutil.rmtree(database_log_path)
        database_log_path.mkdir(parents=True)
        startup_log_file = database_log_path.joinpath("startup.log")
        container_log_thread = ContainerLogThread(db_container,
                                                  self.task_id,
                                                  self.logger,
                                                  startup_log_file,
                                                  "Database Startup %s" % db_container.name)
        container_log_thread.start()
        is_database_ready_thread = IsDatabaseReadyThread(self._database_info, test_container,
                                                         self.db_startup_timeout_in_seconds)
        is_database_ready_thread.start()
        is_database_ready = False
        while (True):
            if container_log_thread.error_message != None:
                is_database_ready = False
                break
            if is_database_ready_thread.finish:
                is_database_ready = True
                break
        container_log_thread.stop()
        is_database_ready_thread.stop()
        container_log_thread.join()
        is_database_ready_thread.join()
        if not is_database_ready:
            if log_config().write_log_files_to_console == WriteLogFilesToConsole.only_error:
                self.logger.error("Task %s: Database startup failed %s failed\nStartup Log:\n%s",
                                  self.task_id,
                                  db_container.name,
                                  "\n".join(container_log_thread.complete_log))
        return is_database_ready

    def save_db_log_files_as_gzip_tar(self, path: pathlib.Path, database_container: Container):
        stream, stat = database_container.get_archive("/exa/logs/db")
        with gzip.open(path, "wb") as file:
            for chunk in stream:
                file.write(chunk)

    def write_output(self):
        with self.output().open("w") as file:
            file.write("READY")
