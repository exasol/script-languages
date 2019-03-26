import logging

import docker
import luigi
import netaddr

from build_utils.lib.build_config import build_config
from build_utils.lib.build_or_pull_db_test_image import BuildOrPullDBTestImage
from build_utils.lib.data.dependency_collector.dependency_container_info_collector import \
    DependencyContainerInfoCollector
from build_utils.lib.data.dependency_collector.dependency_database_info_collector import DependencyDatabaseInfoCollector
from build_utils.lib.data.dependency_collector.dependency_environment_info_collector import ENVIRONMENT_INFO
from build_utils.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from build_utils.lib.data.docker_network_info import DockerNetworkInfo
from build_utils.lib.data.environment_info import EnvironmentInfo
from build_utils.lib.docker_config import docker_config
from build_utils.lib.test_runner.populate_data import PopulateData
from build_utils.lib.test_runner.spawn_test_container import SpawnTestContainer
from build_utils.lib.test_runner.spawn_test_database import SpawnTestDockerDatabase
from build_utils.lib.test_runner.upload_exa_jdbc import UploadExaJDBC
from build_utils.lib.test_runner.upload_virtual_schema_jdbc_adapter import UploadVirtualSchemaJDBCAdapter


class SpawnTestDockerEnvironment(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    environment_name = luigi.Parameter()
    docker_subnet = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._prepare_outputs()
        self._prepare_environment()

    def _prepare_environment(self):
        self.test_container_name = f"""test_container_{self.environment_name}"""
        self.db_container_name = f"""db_container_{self.environment_name}"""
        self.remove_container(self.test_container_name)
        self.remove_container(self.db_container_name)
        self.network_info = self.create_docker_network()
        self.network_info_dict = self.network_info.to_dict()

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

    def requires(self):
        return {
            "database": SpawnTestDockerDatabase(db_container_name=self.db_container_name,
                                                network_info_dict=self.network_info_dict),
            "db_test_image": BuildOrPullDBTestImage()
        }

    def run(self):
        database_info, database_info_dict = self.get_database_info()

        db_test_image_info_dict = self.get_db_test_image_info()

        test_container_info_target = yield SpawnTestContainer(db_test_image_info_dict=db_test_image_info_dict,
                                                              test_container_name=self.test_container_name,
                                                              network_info_dict=self.network_info_dict)
        test_container_info, test_container_info_dict = \
            self.get_test_container_info(test_container_info_target)

        yield [UploadExaJDBC(test_container_info_dict=test_container_info_dict,
                             database_info_dict=database_info_dict),
               UploadVirtualSchemaJDBCAdapter(test_container_info_dict=test_container_info_dict,
                                              database_info_dict=database_info_dict),
               PopulateData(test_container_info_dict=test_container_info_dict,
                            database_info_dict=database_info_dict)]

        self.write_output(database_info, test_container_info)

    def write_output(self, database_info, test_container_info):
        environment_info = \
            EnvironmentInfo(database_info=database_info,
                            test_container_info=test_container_info)
        with self.output()[ENVIRONMENT_INFO].open("w") as file:
            file.write(environment_info.to_json())

    def get_test_container_info(self, test_container_info_target):
        test_container_info = \
            DependencyContainerInfoCollector().get_from_sinlge_input(test_container_info_target)
        test_container_info_dict = test_container_info.to_dict()
        return test_container_info, test_container_info_dict

    def get_database_info(self):
        database_info_of_dependencies = \
            DependencyDatabaseInfoCollector().get_from_dict_of_inputs(self.input())
        database_info = database_info_of_dependencies["database"]
        database_info_dict = database_info.to_dict()
        return database_info, database_info_dict

    def get_db_test_image_info(self):
        image_info_of_dependencies = \
            DependencyImageInfoCollector().get_from_dict_of_inputs(self.input())
        db_image_info = image_info_of_dependencies["db_test_image"]
        db_test_image_info_dict = db_image_info.to_dict()
        return db_test_image_info_dict

    def create_docker_network(self) -> DockerNetworkInfo:
        ip_network = netaddr.IPNetwork(self.docker_subnet)
        network_name = f"""db_network_{self.environment_name}"""
        self.remove_network(network_name)
        subnet = str(ip_network)
        gateway = str(ip_network[1])
        ipam_pool = docker.types.IPAMPool(
            subnet=subnet,
            gateway=gateway
        )
        ipam_config = docker.types.IPAMConfig(
            pool_configs=[ipam_pool]
        )
        network = self._client.networks.create(
            name=network_name,
            driver="bridge",
            ipam=ipam_config
        )
        return DockerNetworkInfo(network_name=network_name, subnet=subnet, gateway=gateway)

    def remove_network(self, network_name):
        try:
            self._client.networks.get(network_name).remove()
            self.logger.info("Remove network %s" % network_name)
        except docker.errors.NotFound:
            pass

    def remove_container(self, container_name: str):
        try:
            container = self._client.containers.get(container_name)
            container.remove(force=True)
            self.logger.info("Removed container %s" % container_name)
        except docker.errors.NotFound:
            pass
