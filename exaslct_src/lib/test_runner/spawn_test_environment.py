import logging
from typing import Dict

import luigi
from luigi import LocalTarget

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_container_info_collector import \
    DependencyContainerInfoCollector
from exaslct_src.lib.data.dependency_collector.dependency_database_info_collector import DependencyDatabaseInfoCollector
from exaslct_src.lib.data.dependency_collector.dependency_docker_network_info_collector import \
    DependencyDockerNetworkInfoCollector
from exaslct_src.lib.data.dependency_collector.dependency_environment_info_collector import ENVIRONMENT_INFO
from exaslct_src.lib.data.environment_info import EnvironmentInfo
from exaslct_src.lib.test_runner.populate_data import PopulateEngineSmallTestDataToDatabase
from exaslct_src.lib.test_runner.prepare_network_for_test_environment import PrepareDockerNetworkForTestEnvironment
from exaslct_src.lib.test_runner.spawn_test_container import SpawnTestContainer
from exaslct_src.lib.test_runner.spawn_test_database import SpawnTestDockerDatabase
from exaslct_src.lib.test_runner.upload_exa_jdbc import UploadExaJDBC
from exaslct_src.lib.test_runner.upload_virtual_schema_jdbc_adapter import UploadVirtualSchemaJDBCAdapter
from exaslct_src.lib.test_runner.wait_for_test_docker_database import WaitForTestDockerDatabase
from exaslct_src.lib.stoppable_task import StoppableTask


class SpawnTestEnvironment(StoppableTask):
    logger = logging.getLogger('luigi-interface')

    environment_name = luigi.Parameter()
    reuse_database_setup = luigi.BoolParameter(False, significant=False)
    reuse_test_container = luigi.BoolParameter(False, significant=False)
    max_start_attempts = luigi.IntParameter(2, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._prepare_outputs()
        self.test_container_name = f"""test_container_{self.environment_name}"""
        self.network_name = f"""db_network_{self.environment_name}"""

    def _prepare_outputs(self):
        self._environment_info_target = luigi.LocalTarget(
            "%s/info/environment/%s/environment_info"
            % (build_config().output_directory,
               self.environment_name))
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
        pass

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
        pass

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
            yield WaitForTestDockerDatabase(environment_name=self.environment_name,
                                            test_container_info_dict=test_container_info_dict,
                                            database_info_dict=database_info_dict,
                                            attempt=attempt)
        with database_ready_target.open("r") as file:
            is_database_ready = file.read() == str(True)
        return is_database_ready

    def setup_test_database(self, test_environment_info_dict):
        # TODO check if database is setup
        yield [UploadExaJDBC(environment_name=self.environment_name,
                             test_environment_info_dict=test_environment_info_dict,
                             reuse_uploaded=self.reuse_database_setup),
               UploadVirtualSchemaJDBCAdapter(
                   environment_name=self.environment_name,
                   test_environment_info_dict=test_environment_info_dict,
                   reuse_uploaded=self.reuse_database_setup),
               PopulateEngineSmallTestDataToDatabase(
                   environment_name=self.environment_name,
                   test_environment_info_dict=test_environment_info_dict,
                   reuse_data=self.reuse_database_setup
               )]

    def write_output(self, environment_info: EnvironmentInfo):
        with self.output()[ENVIRONMENT_INFO].open("w") as file:
            file.write(environment_info.to_json())



class SpawnTestEnvironmentWithDockerDB(SpawnTestEnvironment):

    docker_db_image_name = luigi.OptionalParameter(None)
    docker_db_image_version = luigi.OptionalParameter(None)
    reuse_database = luigi.BoolParameter(False, significant=False)
    database_port_forward = luigi.OptionalParameter(None, significant=False)
    bucketfs_port_forward = luigi.OptionalParameter(None, significant=False)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.db_container_name = f"""db_container_{self.environment_name}"""

    def create_network_task(self, attempt):
        return PrepareDockerNetworkForTestEnvironment(
                environment_name=self.environment_name,
                test_container_name=self.test_container_name,
                db_container_name=self.db_container_name,
                network_name=self.network_name,
                reuse=self.reuse_database,
                attempt=attempt
            )

    def create_spawn_database_task(self, network_info_dict, attempt):
        return SpawnTestDockerDatabase(
                    environment_name=self.environment_name,
                    db_container_name=self.db_container_name,
                    docker_db_image_version = self.docker_db_image_version,
                    docker_db_image_name = self.docker_db_image_name,
                    network_info_dict=network_info_dict,
                    ip_address_index_in_subnet=0,
                    database_port_forward=self.database_port_forward,
                    bucketfs_port_forward=self.bucketfs_port_forward,
                    reuse_database=self.reuse_database,
                    attempt=attempt
                )

class SpawnTestEnvironmentWithExternalDB(SpawnTestEnvironment):

    external_exasol_db_host = luigi.OptionalParameter(None)
    external_exasol_db_port = luigi.OptionalParameter(None)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def create_network_task(self, attempt):
        return 

    def create_spawn_database_task(self, network_info_dict, attempt):
        return
