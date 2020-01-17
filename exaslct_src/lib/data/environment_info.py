from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.data.info import Info


class EnvironmentInfo(Info):

    def __init__(self,
                 name: str,
                 database_info: DatabaseInfo,
                 test_container_info: ContainerInfo):
        self.name = name
        self.test_container_info = test_container_info
        self.database_info = database_info
