import logging

import docker
import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_docker_network_info_collector import DOCKER_NETWORK_INFO
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.stoppable_task import StoppableTask


class PrepareDockerNetworkForTestEnvironment(StoppableTask):
    logger = logging.getLogger('luigi-interface')

    environment_name = luigi.Parameter()
    network_name = luigi.Parameter()
    test_container_name = luigi.Parameter(significant=False)
    db_container_name = luigi.Parameter(significant=False)
    reuse = luigi.BoolParameter(False, significant=False)
    attempt = luigi.IntParameter(-1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = docker_client_config().get_client()
        self._low_level_client = docker_client_config().get_low_level_client()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._network_info_target = luigi.LocalTarget(
            "%s/info/environment/%s/network/%s/%s/network_info"
            % (build_config().output_directory,
               self.environment_name,
               self.network_name,
               self.attempt))
        if self._network_info_target.exists():
            self._network_info_target.remove()

    def output(self):
        return {DOCKER_NETWORK_INFO: self._network_info_target}

    def run_task(self):
        self.network_info = None
        if self.reuse:
            self.logger.info("Task %s: Try to reuse network %s", self.__repr__(), self.network_name)
            try:
                self.network_info = self.reuse_network()
            except Exception as e:
                self.logger.warning("Task %s: Tried to reuse network %s, but got Exeception %s. "
                                    "Fallback to create new network.", self.__repr__(), self.network_name, e)
        if self.network_info is None:
            self.network_info = self.create_docker_network()
        self.write_output(self.network_info)

    def write_output(self, network_info: DockerNetworkInfo):
        with self.output()[DOCKER_NETWORK_INFO].open("w") as file:
            file.write(network_info.to_json())

    def reuse_network(self) -> DockerNetworkInfo:
        self.remove_container(self.test_container_name)
        return self.get_network_info(reused=True)

    def get_network_info(self, reused: bool):
        network_properties = self._low_level_client.inspect_network(self.network_name)
        network_config = network_properties["IPAM"]["Config"][0]
        return DockerNetworkInfo(network_name=self.network_name, subnet=network_config["Subnet"],
                                 gateway=network_config["Gateway"], reused=reused)

    def create_docker_network(self) -> DockerNetworkInfo:
        self.remove_container(self.test_container_name)
        self.remove_container(self.db_container_name)
        self.remove_network(self.network_name)
        network = self._client.networks.create(
            name=self.network_name,
            driver="bridge",
        )
        network_info = self.get_network_info(reused=False)
        subnet = network_info.subnet
        gateway = network_info.gateway
        ipam_pool = docker.types.IPAMPool(
            subnet=subnet,
            gateway=gateway
        )
        ipam_config = docker.types.IPAMConfig(
            pool_configs=[ipam_pool]
        )
        self.remove_network(self.network_name)  # TODO race condition possible, add retry
        network = self._client.networks.create(
            name=self.network_name,
            driver="bridge",
            ipam=ipam_config
        )
        return network_info

    def remove_network(self, network_name):
        try:
            self._client.networks.get(network_name).remove()
            self.logger.info("Task %s: Removed network %s", self.__repr__(), network_name)
        except docker.errors.NotFound:
            pass

    def remove_container(self, container_name: str):
        try:
            container = self._client.containers.get(container_name)
            container.remove(force=True)
            self.logger.info("Task %s: Removed container %s", self.__repr__(), container_name)
        except docker.errors.NotFound:
            pass
