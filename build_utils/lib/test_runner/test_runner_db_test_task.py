import datetime
import logging
import pathlib

import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.data.dependency_collector.dependency_environment_info_collector import \
    DependencyEnvironmentInfoCollector
from build_utils.lib.data.dependency_collector.dependency_release_info_collector import DependencyReleaseInfoCollector
from build_utils.lib.data.environment_info import EnvironmentInfo
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor
from build_utils.lib.test_runner.run_db_tests_in_test_config import RunDBTestsInTestConfig
from build_utils.lib.test_runner.spawn_test_environment import SpawnTestDockerEnvironment
from build_utils.lib.test_runner.upload_release_container import UploadReleaseContainer
from build_utils.stoppable_task import StoppableTask
from build_utils.release_type import ReleaseType


class StopTestEnvironment():
    logger = logging.getLogger('luigi-interface')

    @classmethod
    def stop(cls, test_environment_info: EnvironmentInfo):
        cls.logger.info("Stopping environment %s", test_environment_info.name)
        _docker_config = docker_config()
        _client = docker.DockerClient(base_url=_docker_config.base_url)
        db_container = _client.containers.get(test_environment_info.database_info.container_info.container_name)
        db_container.remove(force=True, v=True)
        test_container = _client.containers.get(test_environment_info.test_container_info.container_name)
        test_container.remove(force=True, v=True)
        network = _client.networks.get(test_environment_info.test_container_info.network_info.network_name)
        network.remove()


class TestRunnerDBTestTask(StoppableTask):
    flavor_path = luigi.Parameter()
    ignore_flavor_test_config = luigi.BoolParameter(False)
    generic_language_tests = luigi.ListParameter([])
    test_folders = luigi.ListParameter([])
    test_files = luigi.ListParameter([])
    tests_to_execute = luigi.ListParameter([])
    languages = luigi.ListParameter([None])
    environment = luigi.DictParameter({"TRAVIS": ""}, significant=False)

    docker_subnet = luigi.Parameter("12.12.12.0/24", significant=False)
    log_level = luigi.Parameter("critical", significant=False)
    reuse_database = luigi.BoolParameter(False, significant=False)
    reuse_uploaded_release_container = luigi.BoolParameter(False, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self.test_environment_info = None
        self._prepare_outputs()

    def _prepare_outputs(self):
        self.flavor_name = flavor.get_name_from_path(str(self.flavor_path))
        self.release_type = self.get_release_type().name
        self.log_path = "%s/logs/test-runner/db-test/tests/%s/%s/%s/" % (
            self._build_config.output_directory,
            self.flavor_name, self.release_type,
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        self._status_target = luigi.LocalTarget(self.log_path + "status.log")
        if self._status_target.exists():
            self._status_target.remove()

    def output(self):
        return self._status_target

    def requires(self):
        test_environment_name = f"""{self.flavor_name}_{self.release_type}"""
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

    def run_task(self):
        release_info = self.get_release_info()
        self.test_environment_info, test_environment_info_dict = self.get_test_environment_info()
        reuse_release_container = self.reuse_database and \
                                  self.reuse_uploaded_release_container and \
                                  not release_info.is_new
        yield UploadReleaseContainer(
            environment_name=self.test_environment_info.name,
            release_name=release_info.name,
            release_type=release_info.release_type.name,
            test_environment_info_dict=test_environment_info_dict,
            release_info_dict=release_info.to_dict(),
            reuse_uploaded=reuse_release_container)

        result_status = yield from self.run_test(test_environment_info_dict)

        with self.output().open("w") as output_file:
            output_file.write(result_status)
        if result_status == "FAILED":
            raise Exception("Some test failed.")

    def run_test(self, test_environment_info_dict):
        test_config = self.read_test_config()
        generic_language_tests = self.get_generic_language_tests(test_config)
        test_folders = self.get_test_folders(test_config)
        test_output = yield RunDBTestsInTestConfig(
            flavor_name=self.flavor_name,
            release_type=self.release_type,
            log_path=self.log_path,
            log_level=self.log_level,
            test_environment_info_dict=test_environment_info_dict,
            environment=self.environment,
            tests_to_execute=self.tests_to_execute,
            generic_language_tests=generic_language_tests,
            test_folders=test_folders,
            language_definition=test_config["language_definition"],
            test_files=self.test_files,
            languages=self.languages
        )
        with test_output.open("r") as test_output_file:
            status = test_output_file.read()
        result_status = "OK"
        for line in status.split("\n"):
            if line != "":
                if line.endswith("FAILED"):
                    result_status = "FAILED"
                    break
        return result_status

    def get_test_folders(self, test_config):
        test_folders = []
        if test_config["test_folders"] != "":
            test_folders = test_config["test_folders"].split(" ")
        if self.ignore_flavor_test_config or len(self.test_folders) != 0:
            test_folders = self.test_folders
        return test_folders

    def get_generic_language_tests(self, test_config):
        generic_language_tests = []
        if test_config["generic_language_tests"] != "":
            generic_language_tests = test_config["generic_language_tests"].split(" ")
        if self.ignore_flavor_test_config or len(self.generic_language_tests) != 0:
            generic_language_tests = self.generic_language_tests
        return generic_language_tests

    def get_release_info(self):
        release_info_of_dependencies = \
            DependencyReleaseInfoCollector().get_from_dict_of_inputs(self.input())
        release_info = release_info_of_dependencies["release"]
        return release_info

    def get_test_environment_info(self):
        test_environment_info_of_dependencies = \
            DependencyEnvironmentInfoCollector().get_from_dict_of_inputs(self.input())
        test_environment_info = test_environment_info_of_dependencies["test_environment"]
        test_environment_info_dict = test_environment_info.to_dict()
        return test_environment_info, test_environment_info_dict

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

    def on_failure(self, exception):
        if not self.reuse_database and self.test_environment_info is not None:
            StopTestEnvironment.stop(self.test_environment_info)
        super().on_failure(exception)

    def on_success(self):
        if not self.reuse_database and self.test_environment_info is not None:
            StopTestEnvironment.stop(self.test_environment_info)
        super().on_success()