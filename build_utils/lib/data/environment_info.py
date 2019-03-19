from build_utils.lib.data.container_info import ContainerInfo
from build_utils.lib.data.database_info import DatabaseInfo
from build_utils.lib.data.info import Info


class EnvironmentInfo(Info):

    def __init__(self, database_info: DatabaseInfo,
                 test_container_info: ContainerInfo):
        self.test_container_info = test_container_info
        self.database_info = database_info
