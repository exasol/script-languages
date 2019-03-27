import pathlib
import time
from typing import Dict, List

import docker
import luigi
from docker.models.containers import Container

from build_utils.lib.build_config import build_config
from build_utils.lib.data.database_info import DatabaseInfo
from build_utils.lib.data.dependency_collector.dependency_environment_info_collector import \
    DependencyEnvironmentInfoCollector
from build_utils.lib.data.dependency_collector.dependency_release_info_collector import DependencyReleaseInfoCollector
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor
from build_utils.lib.test_runner.spawn_test_environment import SpawnTestDockerEnvironment
from build_utils.lib.test_runner.upload_release_container import UploadReleaseContainer
from build_utils.release_type import ReleaseType


class TestRunnerDBTestTask(luigi.Task):
    flavor_path = luigi.Parameter()
    tests_to_execute = luigi.ListParameter([])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._build_config = build_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._log_target = luigi.LocalTarget(
            "%s/logs/test-runner/db-test/%s/%s"
            % (self._build_config.ouput_directory,
               flavor.get_name_from_path(self.flavor_path),
               self.get_release_type().name))
        if self._log_target.exists():
            self._log_target.remove()

    def output(self):
        return self._log_target

    def requires(self):
        test_environment_name = f"""{flavor.get_name_from_path(self.flavor_path)}_{self.get_release_type().name}"""
        return {
            "release": self.get_release_task(self.flavor_path),
            "test_environment": SpawnTestDockerEnvironment(environment_name=test_environment_name,
                                                           docker_subnet="12.12.12.0/24"),
        }

    def get_release_task(self, flavor_path):
        pass

    def get_release_type(self) -> ReleaseType:
        pass

    def run(self):
        release_info_of_dependencies = \
            DependencyReleaseInfoCollector().get_from_dict_of_inputs(self.input())
        release_info = release_info_of_dependencies["release"]

        test_environment_info_of_dependencies = \
            DependencyEnvironmentInfoCollector().get_from_dict_of_inputs(self.input())
        test_environment_info = test_environment_info_of_dependencies["test_environment"]

        test_container_info = test_environment_info.test_container_info
        database_info = test_environment_info.database_info
        yield UploadReleaseContainer(test_container_info_dict=test_container_info.to_dict(),
                                     database_info_dict=database_info.to_dict(),
                                     release_info_dict=release_info.to_dict())

        test_container = self._client.containers.get(test_container_info.container_name)

        test_config = self.read_test_config()
        with self._log_target.open("w") as file:
            self.execute_test(test_container, database_info, test_config,
                              test_config["generic_language_tests"] + "/general.py",
                              ["PythonInterpreter.test_body_is_not_executed_at_creation_time"],
                              file)

    def execute_test(self, test_container: Container, database_info: DatabaseInfo,
                     test_config: Dict[str, str], test: str, tests: List[str], output_file):
        options = "--loglevel=critical " \
                  "--driver=/downloads/ODBC/lib/linux/x86_64/libexaodbc-uo2214lv2.so  " \
                  "--jdbc-path /downloads/JDBC/exajdbc.jar"
        cmd = 'python -tt "{test}" --server "{host}:{port}" --script-languages \'{language}\' {options} {tests}' \
            .format(
            test=test,
            host=database_info.host,
            port=database_info.db_port,
            language=test_config["language_definition"],
            options=options,
            tests=" ".join(tests)
        )
        exit_code, output = test_container.exec_run(cmd=cmd, workdir="/tests/test/", environment={"TRAVIS": ""})
        output_file.write(output.decode("utf-8"))

    def read_test_config(self):
        with pathlib.Path(self.flavor_path).joinpath("testconfig").open("r") as file:
            test_config_str = file.read()
            test_config = {}
            for line in test_config_str.splitlines():
                if not line.startswith("#") and not line == "":
                    split = line.split("=")
                    key = split[0]
                    value = "=".join(split[1:])
                    test_config[key] = value
        return test_config
