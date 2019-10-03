import docker
import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.docker_config import docker_client_config


class PrepareDockerNetworkForTestEnvironment(DependencyLoggerBaseTask):
    environment_name = luigi.Parameter()
    network_name = luigi.Parameter()
    test_container_name = luigi.Parameter(significant=False)
    db_container_name = luigi.OptionalParameter(None, significant=False)
    reuse = luigi.BoolParameter(False, significant=False)
    attempt = luigi.IntParameter(-1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = docker_client_config().get_client()
        self._low_level_client = docker_client_config().get_low_level_client()

    def __del__(self):
        self._client.close()
        self._low_level_client.close()

    def run_task(self):
        self.network_info = None
        if self.reuse:
            self.logger.info("Try to reuse network %s", self.network_name)
            try:
                self.network_info = self.get_network_info(reused=True)
            except Exception as e:
                self.logger.warning("Tried to reuse network %s, but got Exeception %s. "
                                    "Fallback to create new network.", self.network_name, e)
        if self.network_info is None:
            self.network_info = self.create_docker_network()
        self.return_object(self.network_info)

    def get_network_info(self, reused: bool):
        network_properties = self._low_level_client.inspect_network(self.network_name)
        network_config = network_properties["IPAM"]["Config"][0]
        return DockerNetworkInfo(network_name=self.network_name, subnet=network_config["Subnet"],
                                 gateway=network_config["Gateway"], reused=reused)

    def create_docker_network(self) -> DockerNetworkInfo:
        self.remove_container(self.test_container_name)
        if self.db_container_name is not None:
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
            self.logger.info("Removed network %s", network_name)
        except docker.errors.NotFound:
            pass

    def remove_container(self, container_name: str):
        try:
            container = self._client.containers.get(container_name)
            container.remove(force=True)
            self.logger.info("Removed container %s", container_name)
        except docker.errors.NotFound:
            pass
