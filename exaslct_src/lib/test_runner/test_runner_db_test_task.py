import logging
import pathlib
from typing import Generator, Any, Dict

import luigi
from luigi import LocalTarget

from exaslct_src.lib.base.json_pickle_target import JsonPickleTarget
from exaslct_src.lib.data.environment_info import EnvironmentInfo
from exaslct_src.lib.data.release_info import ExportInfo
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.export_containers import ExportFlavorContainer
from exaslct_src.lib.flavor_task import FlavorBaseTask
from exaslct_src.lib.test_runner.database_credentials import DatabaseCredentials
from exaslct_src.lib.test_runner.environment_type import EnvironmentType
from exaslct_src.lib.test_runner.run_db_test_result import RunDBTestsInTestConfigResult
from exaslct_src.lib.test_runner.run_db_tests_in_test_config import RunDBTestsInTestConfig
from exaslct_src.lib.test_runner.run_db_tests_parameter import RunDBTestsInTestConfigParameter
from exaslct_src.lib.test_runner.spawn_test_environment import SpawnTestEnvironment
from exaslct_src.lib.test_runner.spawn_test_environment_parameter import SpawnTestEnvironmentParameter
from exaslct_src.lib.test_runner.upload_exported_container import UploadExportedContainer


class StopTestEnvironment():
    logger = logging.getLogger('luigi-interface')

    @classmethod
    def stop(cls, test_environment_info: EnvironmentInfo):
        cls.logger.info("Stopping environment %s", test_environment_info.name)
        _docker_config = docker_client_config()
        _client = docker_client_config().get_client()
        if test_environment_info.database_info.container_info is not None:
            db_container = _client.containers.get(test_environment_info.database_info.container_info.container_name)
            db_container.remove(force=True, v=True)
        test_container = _client.containers.get(test_environment_info.test_container_info.container_name)
        test_container.remove(force=True, v=True)
        network = _client.networks.get(test_environment_info.test_container_info.network_info.network_name)
        network.remove()


# TODO execute tests only if the exported container is new build
#       - a pulled one is ok,
#       - needs change in image-info and export-info)
#       - add options force tests
#       - only possible if the hash of exaslc also goes into the image hashes
class TestRunnerDBTestTask(FlavorBaseTask,
                           SpawnTestEnvironmentParameter,
                           RunDBTestsInTestConfigParameter):
    reuse_uploaded_container = luigi.BoolParameter(False, significant=False)
    release_goal = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        self.test_environment_info = None
        super().__init__(*args, **kwargs)

    def register_required(self):
        self.register_export_container()
        self.register_spawn_test_environment()

    def register_export_container(self):
        export_container_task = self.create_child_task(ExportFlavorContainer,
                                                       release_goals=[self.release_goal],
                                                       flavor_path=self.flavor_path)
        self._export_infos_future = self.register_dependency(export_container_task)

    def register_spawn_test_environment(self):
        test_environment_name = f"""{self.get_flavor_name()}_{self.release_goal}"""
        spawn_test_environment_task = \
            self.create_child_task_with_common_params(
                SpawnTestEnvironment,
                environment_name=test_environment_name)
        self._test_environment_info_future = self.register_dependency(spawn_test_environment_task)

    def run_task(self):
        export_infos = self.get_values_from_future(self._export_infos_future)  # type: Dict[str,ExportInfo]
        export_info = export_infos[self.release_goal]
        self.test_environment_info = self.get_values_from_future(
            self._test_environment_info_future)  # type: EnvironmentInfo
        reuse_release_container = self.reuse_database and \
                                  self.reuse_uploaded_container and \
                                  not export_info.is_new
        database_credentials = self.get_database_credentials()
        yield from self.upload_container(database_credentials,
                              export_info,
                              reuse_release_container)
        test_results = yield from self.run_test(self.test_environment_info)
        self.return_object(test_results)

    def upload_container(self, database_credentials, export_info, reuse_release_container):
        upload_task = self.create_child_task_with_common_params(
            UploadExportedContainer,
            export_info=export_info,
            environment_name=self.test_environment_info.name,
            test_environment_info=self.test_environment_info,
            release_name=export_info.name,
            reuse_uploaded=reuse_release_container,
            bucketfs_write_password=database_credentials.bucketfs_write_password
        )
        yield from self.run_dependencies(upload_task)

    def get_database_credentials(self) -> DatabaseCredentials:
        if self.environment_type == EnvironmentType.external_db:
            return \
                DatabaseCredentials(db_user=self.external_exasol_db_user,
                                    db_password=self.external_exasol_db_password,
                                    bucketfs_write_password=self.external_exasol_bucketfs_write_password)
        else:
            return \
                DatabaseCredentials(db_user=SpawnTestEnvironment.DEFAULT_DB_USER,
                                    db_password=SpawnTestEnvironment.DEFAULT_DATABASE_PASSWORD,
                                    bucketfs_write_password=SpawnTestEnvironment.DEFAULT_BUCKETFS_WRITE_PASSWORD)

    def run_test(self, test_environment_info: EnvironmentInfo) -> \
            Generator[RunDBTestsInTestConfig, Any, RunDBTestsInTestConfigResult]:
        test_config = self.read_test_config()
        generic_language_tests = self.get_generic_language_tests(test_config)
        test_folders = self.get_test_folders(test_config)
        database_credentials = self.get_database_credentials()
        task = self.create_child_task_with_common_params(
            RunDBTestsInTestConfig,
            test_environment_info=test_environment_info,
            generic_language_tests=generic_language_tests,
            test_folders=test_folders,
            language_definition=test_config["language_definition"],
            db_user=database_credentials.db_user,
            db_password=database_credentials.db_password,
            bucketfs_write_password=database_credentials.bucketfs_write_password
        )
        test_output_future = yield from self.run_dependencies(task)
        test_output = self.get_values_from_future(test_output_future)  # type: RunDBTestsInTestConfigResult
        return test_output

    def get_result_status(self, status):
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
        if self.tests_specified_in_parameters():
            test_folders = self.test_folders
        return test_folders

    def tests_specified_in_parameters(self):
        return len(self.generic_language_tests) != 0 or \
               len(self.test_folders) != 0 or \
               len(self.test_files) != 0

    def get_generic_language_tests(self, test_config):
        generic_language_tests = []
        if test_config["generic_language_tests"] != "":
            generic_language_tests = test_config["generic_language_tests"].split(" ")
        if self.tests_specified_in_parameters():
            generic_language_tests = self.generic_language_tests
        return generic_language_tests

    def read_test_config(self):
        with pathlib.Path(self.flavor_path).joinpath("flavor_base").joinpath("testconfig").open("r") as file:
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
        if not self.reuse_database and \
                self.test_environment_info is not None \
                and isinstance(exception, Exception) \
                and "Some test failed." in str(exception):
            StopTestEnvironment.stop(self.test_environment_info)
        super().on_failure(exception)

    def on_success(self):
        if not self.reuse_database and self.test_environment_info is not None:
            StopTestEnvironment.stop(self.test_environment_info)
        super().on_success()
