from build_utils.lib.data.info import Info


class DockerNetworkInfo(Info):

    def __init__(self, network_name:str, subnet:str, gateway:str):
        self.gateway = gateway
        self.subnet = subnet
        self.network_name = network_name

