import logging

import docker
import luigi
import netaddr

from build_utils.lib.build_config import build_config
from build_utils.lib.data.dependency_collector.dependency_docker_network_info_collector import DOCKER_NETWORK_INFO
from build_utils.lib.data.docker_network_info import DockerNetworkInfo
from build_utils.lib.docker_config import docker_config


class PrepareDockerNetworkForTestEnvironment(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    test_container_name = luigi.Parameter()
    db_container_name = luigi.Parameter()
    network_name = luigi.Parameter()
    reuse = luigi.BoolParameter(False)
    docker_subnet = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._low_level_client = docker.APIClient(base_url=self._docker_config.base_url)
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._network_info_target = luigi.LocalTarget(
            "%s/test-runner/db-test/network/%s/network_info"
            % (self._build_config.ouput_directory,
               self.network_name))
        if self._network_info_target.exists():
            self._network_info_target.remove()

    def output(self):
        return {DOCKER_NETWORK_INFO: self._network_info_target}

    def run(self):
        self.network_info = None
        if self.reuse:
            try:
                self.network_info = self.get_docker_network_info()
            except Exception as e:
                self.logger.warning("Tried to reuse network %s, but got Exeception %s. "
                                    "Fallback to create new network.", self.network_name, e)
        if self.network_info is None:
            self.network_info = self.create_docker_network()
        self.write_output(self.network_info)

    def write_output(self, network_info: DockerNetworkInfo):
        with self.output()[DOCKER_NETWORK_INFO].open("w") as file:
            file.write(network_info.to_json())

    def get_docker_network_info(self) -> DockerNetworkInfo:
        self.remove_container(self.test_container_name)
        network_properties = self._low_level_client.inspect_network(self.network_name)
        network_config = network_properties["IPAM"]["Config"][0]
        return DockerNetworkInfo(network_name=self.network_name, subnet=network_config["Subnet"],
                                 gateway=network_config["Gateway"], reused=True)

    def create_docker_network(self) -> DockerNetworkInfo:
        self.remove_container(self.test_container_name)
        self.remove_container(self.db_container_name)
        ip_network = netaddr.IPNetwork(self.docker_subnet)
        subnet = str(ip_network)
        gateway = str(ip_network[1])
        self.remove_network(self.network_name)
        ipam_pool = docker.types.IPAMPool(
            subnet=subnet,
            gateway=gateway
        )
        ipam_config = docker.types.IPAMConfig(
            pool_configs=[ipam_pool]
        )
        network = self._client.networks.create(
            name=self.network_name,
            driver="bridge",
            ipam=ipam_config
        )
        return DockerNetworkInfo(network_name=self.network_name, subnet=subnet, gateway=gateway)

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
