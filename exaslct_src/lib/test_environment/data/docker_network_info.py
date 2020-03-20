from exaslct_src.lib.base.info import Info


class DockerNetworkInfo(Info):

    def __init__(self, network_name:str, subnet:str, gateway:str, reused:bool=False):
        self.gateway = gateway
        self.subnet = subnet
        self.network_name = network_name
        self.reused = reused

