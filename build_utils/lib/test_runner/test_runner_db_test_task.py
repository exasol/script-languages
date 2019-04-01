import os
import pathlib
import time
from typing import Dict, List

import docker
import luigi
from docker.models.containers import Container

from build_utils.lib.build_config import build_config
from build_utils.lib.data.container_info import ContainerInfo
from build_utils.lib.data.database_info import DatabaseInfo
from build_utils.lib.data.dependency_collector.dependency_environment_info_collector import \
    DependencyEnvironmentInfoCollector
from build_utils.lib.data.dependency_collector.dependency_release_info_collector import DependencyReleaseInfoCollector
from build_utils.lib.data.environment_info import EnvironmentInfo
from build_utils.lib.data.info import FrozenDictToDict
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor
from build_utils.lib.test_runner.spawn_test_environment import SpawnTestDockerEnvironment
from build_utils.lib.test_runner.upload_release_container import UploadReleaseContainer
from build_utils.release_type import ReleaseType


class TestRunnerDBTestTask(luigi.Task):
    flavor_path = luigi.Parameter()
    docker_subnet = luigi.Parameter("12.12.12.0/24")
    tests_to_execute = luigi.ListParameter([])
    log_level = luigi.Parameter("critical")
    environment = luigi.DictParameter({"TRAVIS": ""})
    reuse_database = luigi.BoolParameter(False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._build_config = build_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        flavor_name = flavor.get_name_from_path(str(self.flavor_path))
        path = "%s/logs/test-runner/db-test/%s.log" % (
            self._build_config.ouput_directory,
            flavor_name)
        self._log_target = luigi.LocalTarget(path)
        if self._log_target.exists():
            self._log_target.remove()

    def output(self):
        return self._log_target

    def requires(self):
        test_environment_name = f"""{flavor.get_name_from_path(self.flavor_path)}_{self.get_release_type().name}"""
        return {
            "release": self.get_release_task(self.flavor_path),
            "test_environment": SpawnTestDockerEnvironment(environment_name=test_environment_name,
                                                           docker_subnet=self.docker_subnet,
                                                           reuse_database=self.reuse_database),
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
        test_environment_info_dict = test_environment_info.to_dict()
        yield UploadReleaseContainer(
            test_environment_info_dict=test_environment_info_dict,
            release_info_dict=release_info.to_dict())
        yield RunDBTestDefinedByTestConfig(
            flavor_path=self.flavor_path,
            release_type=self.get_release_type().name,
            test_environment_info_dict=test_environment_info_dict,
            log_level=self.log_level,
            environment=self.environment,
            tests_to_execute=self.tests_to_execute
        )


class RunDBTestDefinedByTestConfig(luigi.Task):
    flavor_path = luigi.Parameter()
    release_type = luigi.Parameter()
    log_level = luigi.Parameter("critical")
    tests_to_execute = luigi.ListParameter([])
    environment = luigi.DictParameter({"TRAVIS": ""})
    test_environment_info_dict = luigi.DictParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._build_config = build_config()
        test_evironment_info = EnvironmentInfo.from_dict(self.test_environment_info_dict)
        self._test_container_info = test_evironment_info.test_container_info
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        flavor_name = flavor.get_name_from_path(str(self.flavor_path))
        path = "%s/logs/test-runner/db-test/%s.log" % (
            self._build_config.ouput_directory,
            flavor_name)
        self._log_target = luigi.LocalTarget(path)
        if self._log_target.exists():
            self._log_target.remove()

    def output(self):
        return self._log_target

    def run(self):
        test_container = self._client.containers.get(self._test_container_info.container_name)
        test_config = self.read_test_config()
        flavor_name = flavor.get_name_from_path(str(self.flavor_path))
        log_path = pathlib.Path(flavor_name).joinpath(str(self.release_type))
        language_definition = test_config["language_definition"]
        with self.output().open("w") as file:
            for generic_language in test_config["generic_language_tests"].split(" "):
                for test_file, test_task_config in \
                        self.generate_test_task_configs_from_directory(
                            test_container, generic_language, language_definition, log_path):
                    test_task_config["language"] = generic_language
                    test_output = yield RunDBTest(**test_task_config)
                    with test_output.open("r") as test_output_file:
                        exit_code = test_output_file.read()
                    file.write("%s %s %s\n" % (generic_language, test_file, exit_code))

    def generate_test_task_configs_from_directory(
            self, test_container: Container, directory: str,
            language_definition: str, log_path: pathlib.Path):
        exit_code, ls_output = test_container.exec_run(cmd="ls /tests/test/%s/" % directory)
        test_files = ls_output.decode("utf-8").split("\n")
        for test_file in test_files:
            if test_file != "":
                config = dict(test_evironment_info_dict=self.test_environment_info_dict,
                              language_definition=language_definition,
                              log_level=self.log_level,
                              environment=self.environment,
                              log_path=str(log_path),
                              language=directory,
                              test_file=directory + "/" + test_file)
                yield test_file, config

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


class RunDBTest(luigi.Task):
    test_evironment_info_dict = luigi.DictParameter()
    log_level = luigi.Parameter()
    test_file = luigi.Parameter()
    tests_to_execute = luigi.ListParameter([])
    language_definition = luigi.Parameter()
    log_path = luigi.Parameter()
    environment = luigi.DictParameter({"TRAVIS": ""})
    language = luigi.OptionalParameter("")

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
        path = "%s/logs/test-runner/db-test/%s/%s" % (
            self._build_config.ouput_directory, self.log_path, self.test_file)
        self._log_target = pathlib.Path(path + "/log")
        if self._log_target.exists():
            os.remove(self._log_target)
        self._exit_code_target = luigi.LocalTarget(path + "/exit_code")
        if self._exit_code_target.exists():
            self._exit_code_target.remove()

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
        environment = FrozenDictToDict().convert(self.environment)
        exit_code, output = test_container.exec_run(cmd=cmd, workdir="/tests/test/",
                                                    environment=environment)
        self._log_target.parent.mkdir(parents=True, exist_ok=True)
        with self._log_target.open("w") as file:
            file.write(output.decode("utf-8"))
        with self.output().open("w") as file:
            file.write(str(exit_code))
