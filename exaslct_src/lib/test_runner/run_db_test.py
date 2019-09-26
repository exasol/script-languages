import logging
import pathlib

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.environment_info import EnvironmentInfo
from exaslct_src.lib.data.info import FrozenDictToDict
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.log_config import log_config, WriteLogFilesToConsole
from exaslct_src.lib.still_running_logger import StillRunningLogger, StillRunningLoggerThread
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.test_runner.database_credentials import DatabaseCredentialsParameter


class RunDBTest(StoppableTask, DatabaseCredentialsParameter):
    logger = logging.getLogger('luigi-interface')

    test_file = luigi.Parameter()
    flavor_name = luigi.Parameter()
    release_type = luigi.Parameter()
    language = luigi.OptionalParameter("")
    test_restrictions = luigi.ListParameter([])
    test_environment_vars = luigi.DictParameter({"TRAVIS": ""}, significant=False)
    language_definition = luigi.Parameter(significant=False)

    log_path = luigi.Parameter(significant=False)
    log_level = luigi.Parameter(significant=False)
    test_environment_info_dict = luigi.DictParameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        test_evironment_info = EnvironmentInfo.from_dict(self.test_environment_info_dict)
        self._test_container_info = test_evironment_info.test_container_info
        self._database_info = test_evironment_info.database_info
        self._client = docker_client_config().get_client()
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        test_file_name = pathlib.Path(self.test_file).name
        path = pathlib.Path(self.log_path).joinpath(test_file_name)
        self._log_target = path.joinpath("log")
        # if self._log_target.exists():
        #     os.remove(self._log_target)
        status_path = path.joinpath("status.log")
        self._status_target = luigi.LocalTarget(str(status_path))
        # if self._exit_code_target.exists():
        #     self._exit_code_target.remove()

    def output(self):
        return self._status_target

    def run_task(self):
        self.logger.info("Task %s: Running db tests of flavor %s and release %s in %s"
                         % (self.__repr__(), self.flavor_name, self.release_type, self.test_file))
        test_container = self._client.containers.get(self._test_container_info.container_name)
        bash_cmd = self.generate_test_command()
        environment, exit_code, output = self.run_test_command(bash_cmd, test_container)
        self.handle_test_result(bash_cmd, environment, exit_code, output)

    def handle_test_result(self, bash_cmd, environment, exit_code, output):
        self._log_target.parent.mkdir(parents=True, exist_ok=True)
        log_output = "command: " + bash_cmd + "\n" + \
                     "environment: " + str(environment) + "\n" + \
                     output.decode("utf-8")
        if log_config().write_log_files_to_console == WriteLogFilesToConsole.all:
            self.logger.info("Task %s: Test results for db tests of flavor %s and release %s in %s\n%s"
                             % (self.__repr__(), self.flavor_name, self.release_type, self.test_file, log_output))
        if log_config().write_log_files_to_console == WriteLogFilesToConsole.only_error and exit_code != 0:
            self.logger.error("Task %s: db tests of flavor %s and release %s in %s failed\nTest results:\n%s"
                              % (self.__repr__(), self.flavor_name, self.release_type, self.test_file, log_output))
        with self._log_target.open("w") as file:
            file.write(log_output)
        with self.output().open("w") as file:
            if exit_code == 0:
                file.write("OK")
            else:
                file.write("FAILED")

    def run_test_command(self, bash_cmd, test_container):
        still_running_logger = StillRunningLogger(self.logger, self.__repr__(),
                                                  "db tests of flavor %s and release %s in %s"
                                                  % (self.flavor_name, self.release_type, self.test_file))
        thread = StillRunningLoggerThread(still_running_logger)
        thread.start()
        environment = FrozenDictToDict().convert(self.test_environment_vars)
        exit_code, output = test_container.exec_run(cmd=bash_cmd,
                                                    environment=environment)
        thread.stop()
        thread.join()
        return environment, exit_code, output

    def generate_test_command(self):
        credentials = f"--user '{self.db_user}' --password '{self.db_password}'"
        log_level = f"--loglevel={self.log_level}"
        server = f"--server '{self._database_info.host}:{self._database_info.db_port}'"
        environment = "--driver=/downloads/ODBC/lib/linux/x86_64/libexaodbc-uo2214lv2.so  " \
                      "--jdbc-path /downloads/JDBC/exajdbc.jar"
        language_definition = f"--script-languages '{self.language_definition}'"
        language = ""
        if self.language is not None:
            language = "--lang %s" % self.language
        test_restrictions = " ".join(self.test_restrictions)
        test_file = f'"{self.test_file}"'
        args = " ".join([test_file,
                         server,
                         credentials,
                         language_definition,
                         log_level,
                         environment,
                         language,
                         test_restrictions])
        cmd = f'cd /tests/test/; python -tt {args}'
        bash_cmd = f"""bash -c "{cmd}" """
        return bash_cmd
