from build_utils.lib.data.docker_network_info import DockerNetworkInfo
from build_utils.lib.data.info import Info


class ContainerInfo(Info):

    def __init__(self, container_name:str, network_info:DockerNetworkInfo=None, volume_name:str=None):
        self.network_info = network_info
        self.container_name = container_name
        self.volume_name = volume_name
