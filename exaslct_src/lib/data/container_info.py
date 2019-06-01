from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.data.info import Info


class ContainerInfo(Info):

    def __init__(self, container_name: str,
                 ip_address: str, network_info: DockerNetworkInfo,
                 volume_name: str = None):
        self.ip_address = ip_address
        self.network_info = network_info
        self.container_name = container_name
        self.volume_name = volume_name
