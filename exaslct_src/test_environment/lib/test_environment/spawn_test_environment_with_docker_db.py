from exaslct_src.test_environment.lib.test_environment.abstract_spawn_test_environment import AbstractSpawnTestEnvironment
from exaslct_src.test_environment.lib.data.container_info import ContainerInfo
from exaslct_src.test_environment.lib.data.database_info import DatabaseInfo
from exaslct_src.test_environment.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.test_environment.lib.data.environment_type import EnvironmentType
from exaslct_src.test_environment.lib.test_environment.docker_db_test_environment_parameter import DockerDBTestEnvironmentParameter
from exaslct_src.test_environment.lib.test_environment.prepare_network_for_test_environment import PrepareDockerNetworkForTestEnvironment
from exaslct_src.test_environment.lib.test_environment.spawn_test_database import SpawnTestDockerDatabase
from exaslct_src.test_environment.lib.test_environment.wait_for_test_docker_database import WaitForTestDockerDatabase


class SpawnTestEnvironmentWithDockerDB(
    AbstractSpawnTestEnvironment,
    DockerDBTestEnvironmentParameter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_container_name = f"""db_container_{self.environment_name}"""

    def get_environment_type(self):
        return EnvironmentType.docker_db

    def create_network_task(self, attempt: int):
        return \
            self.create_child_task_with_common_params(
                PrepareDockerNetworkForTestEnvironment,
                test_container_name=self.test_container_name,
                network_name=self.network_name,
                db_container_name=self.db_container_name,
                reuse=self.reuse_database or self.reuse_test_container,
                no_cleanup_after_end=self.no_database_cleanup_after_end or self.no_test_container_cleanup_after_end,
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
