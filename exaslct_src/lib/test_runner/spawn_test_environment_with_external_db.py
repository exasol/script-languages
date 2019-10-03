import luigi

from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.test_runner.determine_external_database_host import DetermineExternalDatabaseHost
from exaslct_src.lib.test_runner.external_test_environment_parameter import ExternalTestEnvironmentParameter, \
    ExternalDatabaseHostParameter
from exaslct_src.lib.test_runner.prepare_network_for_test_environment import PrepareDockerNetworkForTestEnvironment
from exaslct_src.lib.test_runner.abstract_spawn_test_environment import AbstractSpawnTestEnvironment
from exaslct_src.lib.test_runner.wait_for_external_database import WaitForTestExternalDatabase


class SpawnTestEnvironmentWithExternalDB(AbstractSpawnTestEnvironment,
                                         ExternalDatabaseHostParameter):

    def create_network_task(self, attempt: int):
        return \
            self.create_child_task_with_common_params(
                PrepareDockerNetworkForTestEnvironment,
                reuse=self.reuse_test_container,
                attempt=attempt,
                test_container_name=self.test_container_name,
                network_name=self.network_name
            )

    def create_spawn_database_task(self, network_info: DockerNetworkInfo, attempt: int):
        return \
            self.create_child_task_with_common_params(
                DetermineExternalDatabaseHost,
                network_info=network_info,
                attempt=attempt
            )

    def create_wait_for_database_task(self,
                                      attempt: int,
                                      database_info: DatabaseInfo,
                                      test_container_info: ContainerInfo):
        return \
            self.create_child_task_with_common_params(
                WaitForTestExternalDatabase,
                test_container_info=test_container_info,
                database_info=database_info,
                attempt=attempt)
