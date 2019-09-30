import logging
from typing import Dict

import luigi
from luigi import LocalTarget

from exaslct_src.AbstractMethodException import AbstractMethodException
from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_container_info_collector import \
    DependencyContainerInfoCollector
from exaslct_src.lib.data.dependency_collector.dependency_database_info_collector import DependencyDatabaseInfoCollector
from exaslct_src.lib.data.dependency_collector.dependency_docker_network_info_collector import \
    DependencyDockerNetworkInfoCollector
from exaslct_src.lib.data.dependency_collector.dependency_environment_info_collector import ENVIRONMENT_INFO
from exaslct_src.lib.data.environment_info import EnvironmentInfo
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.test_runner.database_credentials import DatabaseCredentialsParameter
from exaslct_src.lib.test_runner.general_spawn_test_environment_parameter import GeneralSpawnTestEnvironmentParameter
from exaslct_src.lib.test_runner.populate_data import PopulateEngineSmallTestDataToDatabase
from exaslct_src.lib.test_runner.spawn_test_container import SpawnTestContainer
from exaslct_src.lib.test_runner.upload_exa_jdbc import UploadExaJDBC
from exaslct_src.lib.test_runner.upload_virtual_schema_jdbc_adapter import UploadVirtualSchemaJDBCAdapter


class AbstractSpawnTestEnvironment(StoppableTask,
                                   GeneralSpawnTestEnvironmentParameter,
                                   DatabaseCredentialsParameter):
    logger = logging.getLogger('luigi-interface')

    environment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._prepare_outputs()
        self.test_container_name = f"""test_container_{self.environment_name}"""
        self.network_name = f"""db_network_{self.environment_name}"""

    def _prepare_outputs(self):
        self._environment_info_target = luigi.LocalTarget(
            "%s/info/environment/%s/%s/environment_info"
            % (build_config().output_directory,
               self.environment_name,
               self.__class__.__name__))
        if self._environment_info_target.exists():
            self._environment_info_target.remove()

    def output(self):
        return {
            ENVIRONMENT_INFO: self._environment_info_target,
        }

    def run_task(self):
        test_environment_info = yield from self.attempt_database_start()
        test_environment_info_dict = test_environment_info.to_dict()
        yield from self.setup_test_database(test_environment_info_dict)
        self.write_output(test_environment_info)

    def attempt_database_start(self):
        is_database_ready = False
        attempt = 0
        while not is_database_ready and attempt < self.max_start_attempts:
            database_info, is_database_ready, test_container_info = \
                yield from self.start_database(attempt)
            attempt += 1
        if not is_database_ready and not attempt < self.max_start_attempts:
            raise Exception(f"Maximum attempts {attempt} to start the database reached.")
        test_environment_info = \
            EnvironmentInfo(name=self.environment_name,
                            database_info=database_info,
                            test_container_info=test_container_info)
        return test_environment_info

    def start_database(self, attempt):
        network_info_dict = yield from self.create_network(attempt)
        database_info, database_info_dict, \
        test_container_info, test_container_info_dict = \
            yield from self.spawn_database_and_test_container(network_info_dict, attempt)
        is_database_ready = yield from self.wait_for_database(database_info_dict, test_container_info_dict, attempt)
        return database_info, is_database_ready, test_container_info

    def create_network(self, attempt):
        docker_network_output = yield self.create_network_task(attempt)
        network_info, network_info_dict = \
            self.get_network_info(docker_network_output)
        return network_info_dict

    def create_network_task(self, attempt):
        raise AbstractMethodException()

    def get_network_info(self, network_info_target):
        network_info = \
            DependencyDockerNetworkInfoCollector().get_from_sinlge_input(network_info_target)
        network_info_dict = network_info.to_dict()
        return network_info, network_info_dict

    def spawn_database_and_test_container(self, network_info_dict, attempt):
        database_and_test_container_output = \
            yield {
                "test_container": SpawnTestContainer(
                    environment_name=self.environment_name,
                    test_container_name=self.test_container_name,
                    network_info_dict=network_info_dict,
                    ip_address_index_in_subnet=1,
                    reuse_test_container=self.reuse_test_container,
                    attempt=attempt),
                "database": self.create_spawn_database_task(network_info_dict, attempt)
            }
        test_container_info, test_container_info_dict = \
            self.get_test_container_info(database_and_test_container_output)
        database_info, database_info_dict = \
            self.get_database_info(database_and_test_container_output)
        return database_info, database_info_dict, \
               test_container_info, test_container_info_dict

    def create_spawn_database_task(self, network_info_dict, attempt):
        raise AbstractMethodException()

    def get_test_container_info(self, input: Dict[str, Dict[str, LocalTarget]]):
        container_info_of_dependencies = \
            DependencyContainerInfoCollector().get_from_dict_of_inputs(input)
        test_container_info = container_info_of_dependencies["test_container"]
        test_container_info_dict = test_container_info.to_dict()
        return test_container_info, test_container_info_dict

    def get_database_info(self, input: Dict[str, Dict[str, LocalTarget]]):
        database_info_of_dependencies = \
            DependencyDatabaseInfoCollector().get_from_dict_of_inputs(input)
        database_info = database_info_of_dependencies["database"]
        database_info_dict = database_info.to_dict()
        return database_info, database_info_dict

    def wait_for_database(self, database_info_dict, test_container_info_dict, attempt):
        database_ready_target = \
            yield self.create_wait_for_database_task(attempt, database_info_dict, test_container_info_dict)
        with database_ready_target.open("r") as file:
            wait_status = file.read()
            is_database_ready = wait_status == str(True)
        return is_database_ready

    def create_wait_for_database_task(self, attempt, database_info_dict, test_container_info_dict):
        raise AbstractMethodException()

    def setup_test_database(self, test_environment_info_dict):
        # TODO check if database is setup
        self.logger.info("Task %s: Setup database",self.__repr__())
        yield [
            UploadExaJDBC(
                environment_name=self.environment_name,
                test_environment_info_dict=test_environment_info_dict,
                reuse_uploaded=self.reuse_database_setup,
                bucketfs_write_password=self.bucketfs_write_password),
            UploadVirtualSchemaJDBCAdapter(
                environment_name=self.environment_name,
                test_environment_info_dict=test_environment_info_dict,
                reuse_uploaded=self.reuse_database_setup,
                bucketfs_write_password=self.bucketfs_write_password),
            PopulateEngineSmallTestDataToDatabase(
                environment_name=self.environment_name,
                test_environment_info_dict=test_environment_info_dict,
                reuse_data=self.reuse_database_setup,
                db_user=self.db_user,
                db_password=self.db_password,
                bucketfs_write_password=self.bucketfs_write_password
            )]

    def write_output(self, environment_info: EnvironmentInfo):
        with self.output()[ENVIRONMENT_INFO].open("w") as file:
            file.write(environment_info.to_json())
