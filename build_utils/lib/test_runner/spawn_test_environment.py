import logging
from typing import Dict

import luigi
from luigi import LocalTarget

from build_utils.lib.build_config import build_config
from build_utils.lib.build_or_pull_db_test_image import BuildOrPullDBTestImage
from build_utils.lib.data.dependency_collector.dependency_container_info_collector import \
    DependencyContainerInfoCollector
from build_utils.lib.data.dependency_collector.dependency_database_info_collector import DependencyDatabaseInfoCollector
from build_utils.lib.data.dependency_collector.dependency_docker_network_info_collector import \
    DependencyDockerNetworkInfoCollector
from build_utils.lib.data.dependency_collector.dependency_environment_info_collector import ENVIRONMENT_INFO
from build_utils.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from build_utils.lib.data.environment_info import EnvironmentInfo
from build_utils.lib.test_runner.populate_data import PopulateData
from build_utils.lib.test_runner.prepare_network_for_test_environment import PrepareDockerNetworkForTestEnvironment
from build_utils.lib.test_runner.spawn_test_container import SpawnTestContainer
from build_utils.lib.test_runner.spawn_test_database import SpawnTestDockerDatabase
from build_utils.lib.test_runner.upload_exa_jdbc import UploadExaJDBC
from build_utils.lib.test_runner.upload_virtual_schema_jdbc_adapter import UploadVirtualSchemaJDBCAdapter


class SpawnTestDockerEnvironment(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    reuse_database = luigi.BoolParameter(False)
    environment_name = luigi.Parameter()
    docker_subnet = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._prepare_outputs()
        self.test_container_name = f"""test_container_{self.environment_name}"""
        self.db_container_name = f"""db_container_{self.environment_name}"""
        self.network_name = f"""db_network_{self.environment_name}"""

    def _prepare_outputs(self):
        self._environment_info_target = luigi.LocalTarget(
            "%s/test-runner/db-test/test-environment/%s/environment_info"
            % (self._build_config.ouput_directory,
               self.environment_name))
        if self._environment_info_target.exists():
            self._environment_info_target.remove()

    def output(self):
        return {
            ENVIRONMENT_INFO: self._environment_info_target,
        }

    def run(self):
        docker_network_output = yield PrepareDockerNetworkForTestEnvironment(
            test_container_name=self.test_container_name,
            db_container_name=self.db_container_name,
            network_name=self.network_name,
            reuse=self.reuse_database,
            docker_subnet=self.docker_subnet
        )
        network_info, network_info_dict = self.get_network_container_info(docker_network_output)
        database_and_db_test_image_output = \
            yield {
                "database": SpawnTestDockerDatabase(db_container_name=self.db_container_name,
                                                    network_info_dict=network_info_dict),
                "db_test_image": BuildOrPullDBTestImage()
            }
        database_info, database_info_dict = self.get_database_info(database_and_db_test_image_output)
        db_test_image_info_dict = self.get_db_test_image_info(database_and_db_test_image_output)

        test_container_info_target = \
            yield SpawnTestContainer(db_test_image_info_dict=db_test_image_info_dict,
                                     test_container_name=self.test_container_name,
                                     network_info_dict=network_info_dict)
        test_container_info, test_container_info_dict = \
            self.get_test_container_info(test_container_info_target)
        test_environment_info = \
            EnvironmentInfo(database_info=database_info,
                            test_container_info=test_container_info)
        test_environment_info_dict = test_environment_info.to_dict()
        yield [UploadExaJDBC(test_environment_info_dict=test_environment_info_dict),
               UploadVirtualSchemaJDBCAdapter(test_environment_info_dict=test_environment_info_dict),
               PopulateData(test_environment_info_dict=test_environment_info_dict)]

        self.write_output(test_environment_info)

    def write_output(self, environment_info: EnvironmentInfo):
        with self.output()[ENVIRONMENT_INFO].open("w") as file:
            file.write(environment_info.to_json())

    def get_network_container_info(self, network_info_target):
        network_info = \
            DependencyDockerNetworkInfoCollector().get_from_sinlge_input(network_info_target)
        network_info_dict = network_info.to_dict()
        return network_info, network_info_dict

    def get_test_container_info(self, test_container_info_target):
        test_container_info = \
            DependencyContainerInfoCollector().get_from_sinlge_input(test_container_info_target)
        test_container_info_dict = test_container_info.to_dict()
        return test_container_info, test_container_info_dict

    def get_database_info(self, input: Dict[str, Dict[str, LocalTarget]]):
        database_info_of_dependencies = \
            DependencyDatabaseInfoCollector().get_from_dict_of_inputs(input)
        database_info = database_info_of_dependencies["database"]
        database_info_dict = database_info.to_dict()
        return database_info, database_info_dict

    def get_db_test_image_info(self, input: Dict[str, Dict[str, LocalTarget]]):
        image_info_of_dependencies = \
            DependencyImageInfoCollector().get_from_dict_of_inputs(input)
        db_image_info = image_info_of_dependencies["db_test_image"]
        db_test_image_info_dict = db_image_info.to_dict()
        return db_test_image_info_dict
