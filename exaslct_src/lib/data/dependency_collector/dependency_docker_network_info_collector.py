from typing import Dict

from exaslct_src.lib.data.container_info import DockerNetworkInfo
from exaslct_src.lib.data.dependency_collector.dependency_collector import DependencyInfoCollector


class DependencyDockerNetworkInfoCollector(DependencyInfoCollector[DockerNetworkInfo]):

    def is_info(self, input):
        return isinstance(input, Dict) and DOCKER_NETWORK_INFO in input

    def read_info(self, value) -> DockerNetworkInfo:
        with value[DOCKER_NETWORK_INFO].open("r") as file:
            return DockerNetworkInfo.from_json(file.read())


DOCKER_NETWORK_INFO = "docker_network_info"
