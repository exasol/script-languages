import luigi

from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.test_runner.docker_db_test_environment_parameter import DockerDBTestEnvironmentParameter
from exaslct_src.lib.test_runner.prepare_network_for_test_environment import PrepareDockerNetworkForTestEnvironment
from exaslct_src.lib.test_runner.spawn_test_database import SpawnTestDockerDatabase
from exaslct_src.lib.test_runner.abstract_spawn_test_environment import AbstractSpawnTestEnvironment
from exaslct_src.lib.test_runner.wait_for_test_docker_database import WaitForTestDockerDatabase


class SpawnTestEnvironmentWithDockerDB(
    AbstractSpawnTestEnvironment,
    DockerDBTestEnvironmentParameter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_container_name = f"""db_container_{self.environment_name}"""

    def create_network_task(self, attempt: int):
        return \
            self.create_child_task_with_common_params(
                PrepareDockerNetworkForTestEnvironment,
                test_container_name=self.test_container_name,
                network_name=self.network_name,
                db_container_name=self.db_container_name,
                reuse=self.reuse_database,
                attempt=attempt
            )

    def create_spawn_database_task(self,
                                   network_info: DockerNetworkInfo,
                                   attempt: int):
        return \
            self.create_child_task_with_common_params(
                SpawnTestDockerDatabase,
                db_container_name=self.db_container_name,
                network_info=network_info,
                ip_address_index_in_subnet=0,
                attempt=attempt
            )

    def create_wait_for_database_task(self,
                                      attempt: int,
                                      database_info: DatabaseInfo,
                                      test_container_info: ContainerInfo):
        return \
            self.create_child_task_with_common_params(
                WaitForTestDockerDatabase,
                test_container_info=test_container_info,
                database_info=database_info,
                attempt=attempt)
