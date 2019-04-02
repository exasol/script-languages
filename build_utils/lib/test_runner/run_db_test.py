import logging
import os
import pathlib

import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.data.environment_info import EnvironmentInfo
from build_utils.lib.data.info import FrozenDictToDict
from build_utils.lib.docker_config import docker_config


class RunDBTest(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    test_file = luigi.Parameter()
    tests_to_execute = luigi.ListParameter([])
    language = luigi.OptionalParameter("")
    environment = luigi.DictParameter({"TRAVIS": ""})
    language_definition = luigi.Parameter()

    log_path = luigi.Parameter()
    log_level = luigi.Parameter()
    test_evironment_info_dict = luigi.DictParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._build_config = build_config()
        test_evironment_info = EnvironmentInfo.from_dict(self.test_evironment_info_dict)
        self._test_container_info = test_evironment_info.test_container_info
        self._database_info = test_evironment_info.database_info
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        test_file_name = pathlib.Path(self.test_file).name
        path = pathlib.Path(self.log_path).joinpath(test_file_name)
        self._log_target = path.joinpath("log")
        # if self._log_target.exists():
        #     os.remove(self._log_target)
        exit_code_path = path.joinpath("exit_code")
        self._exit_code_target = luigi.LocalTarget(str(exit_code_path))
        # if self._exit_code_target.exists():
        #     self._exit_code_target.remove()

    def output(self):
        return self._exit_code_target

    def run(self):
        test_container = self._client.containers.get(self._test_container_info.container_name)
        log_level = "--loglevel=%s" % self.log_level
        server = '--server "%s:%s"' % (self._database_info.host, self._database_info.db_port)
        environment = "--driver=/downloads/ODBC/lib/linux/x86_64/libexaodbc-uo2214lv2.so  " \
                      "--jdbc-path /downloads/JDBC/exajdbc.jar"
        language_definition = "--script-languages '%s'" % self.language_definition
        language = ""
        if self.language is not None:
            language = "--lang %s" % self.language
        args = '"{test_file}" {server} {language_definition} {log_level} {environment} {language} {tests}' \
            .format(
            test_file=self.test_file,
            server=server,
            language_definition=language_definition,
            log_level=log_level,
            environment=environment,
            language=language,
            tests=" ".join(self.tests_to_execute)
        )
        cmd = 'python -tt %s' % args
        self.logger.debug("%s: Executing test command %s", self.task_id, cmd)
        environment = FrozenDictToDict().convert(self.environment)
        exit_code, output = test_container.exec_run(cmd=cmd, workdir="/tests/test/",
                                                    environment=environment)
        self._log_target.parent.mkdir(parents=True, exist_ok=True)
        with self._log_target.open("w") as file:
            file.write(output.decode("utf-8"))
        with self.output().open("w") as file:
            file.write(str(exit_code))
