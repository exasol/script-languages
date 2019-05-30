import logging

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.environment_info import EnvironmentInfo
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.stoppable_task import StoppableTask


class PopulateEngineSmallTestDataToDatabase(StoppableTask):
    logger = logging.getLogger('luigi-interface')

    environment_name = luigi.Parameter()
    reuse_data = luigi.BoolParameter(False, significant=False)
    test_environment_info_dict = luigi.DictParameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._client = docker_client_config().get_client()

        self._test_environment_info = EnvironmentInfo.from_dict(self.test_environment_info_dict)
        self._test_container_info = self._test_environment_info.test_container_info
        self._database_info = self._test_environment_info.database_info
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._log_target = luigi.LocalTarget(
            "%s/logs/environment/%s/populate_data/%s"
            % (build_config().output_directory,
               self._test_environment_info.name,
               self.task_id))
        if self._log_target.exists():
            self._log_target.remove()

    def output(self):
        return self._log_target

    def run_task(self):
        if not self.reuse_data:
            self.populate_data()
        else:
            self.logger.warning("Task %s: Reusing data", self.__repr__())
            self.write_logs("Reused")

    def populate_data(self):
        self.logger.warning("Task %s: Uploading data", self.__repr__())
        username = "sys"
        password = "exasol"
        test_container = self._client.containers.get(self._test_container_info.container_name)
        cmd = f"""cd /tests/test/enginedb_small; $EXAPLUS -c '{self._database_info.host}:{self._database_info.db_port}' -u '{username}' -p '{password}' -f import.sql"""
        bash_cmd = f"""bash -c "{cmd}" """
        exit_code, output = test_container.exec_run(cmd=bash_cmd)
        self.write_logs(output.decode("utf-8"))
        if exit_code != 0:
            raise Exception("Failed to populate the database with data.\nLog: %s" % cmd + "\n" + output.decode("utf-8"))

    def write_logs(self, output: str):
        with self._log_target.open("w") as file:
            file.write(output)
