from typing import List

from exaslct_src.test_environment.base.info import Info
from exaslct_src.test_environment.lib.data.docker_network_info import DockerNetworkInfo


class ContainerInfo(Info):

    def __init__(self, container_name: str,
                 ip_address: str,
                 network_aliases: List[str],
                 network_info: DockerNetworkInfo,
                 volume_name: str = None):
        self.network_aliases = network_aliases
        self.ip_address = ip_address
        self.network_info = network_info
        self.container_name = container_name
        self.volume_name = volume_name
