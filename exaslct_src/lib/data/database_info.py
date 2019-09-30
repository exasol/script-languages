from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.info import Info
from exaslct_src.lib.test_runner.environment_type import EnvironmentType


class DatabaseInfo(Info):

    def __init__(self, host: str, db_port: str, bucketfs_port: str,
                 container_info: ContainerInfo = None):
        self.container_info = container_info
        self.bucketfs_port = bucketfs_port
        self.db_port = db_port
        self.host = host
