from build_utils.lib.data.container_info import ContainerInfo
from build_utils.lib.data.info import Info


class DatabaseInfo(Info):

    def __init__(self, host: str, db_port: str, bucketfs_port: str, container_info: ContainerInfo = None):
        self.container_info = container_info
        self.bucketfs_port = bucketfs_port
        self.db_port = db_port
        self.host = host

