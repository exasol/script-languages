from typing import Dict

from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.dependency_collector.dependency_collector import DependencyInfoCollector


class DependencyContainerInfoCollector(DependencyInfoCollector[ContainerInfo]):

    def is_info(self, input):
        return isinstance(input, Dict) and CONTAINER_INFO in input

    def read_info(self, value) -> ContainerInfo:
        with value[CONTAINER_INFO].open("r") as file:
            return ContainerInfo.from_json(file.read())


CONTAINER_INFO = "container_info"
