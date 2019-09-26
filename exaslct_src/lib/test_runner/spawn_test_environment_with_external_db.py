import luigi

from exaslct_src.lib.test_runner.determine_external_database_host import DetermineExternalDatabaseHost
from exaslct_src.lib.test_runner.external_test_environment_parameter import ExternalTestEnvironmentParameter, \
    ExternalDatabaseHostParameter
from exaslct_src.lib.test_runner.prepare_network_for_test_environment import PrepareDockerNetworkForTestEnvironment
from exaslct_src.lib.test_runner.abstract_spawn_test_environment import AbstractSpawnTestEnvironment
from exaslct_src.lib.test_runner.wait_for_external_database import WaitForTestExternalDatabase


class SpawnTestEnvironmentWithExternalDB(AbstractSpawnTestEnvironment,
                                         ExternalDatabaseHostParameter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_network_task(self, attempt):
        return \
            PrepareDockerNetworkForTestEnvironment(
                environment_name=self.environment_name,
                test_container_name=self.test_container_name,
                network_name=self.network_name,
                reuse=self.reuse_test_container,
                attempt=attempt
            )

    def create_spawn_database_task(self, network_info_dict, attempt):
        return \
            DetermineExternalDatabaseHost(
                environment_name=self.environment_name,
                external_exasol_db_host=self.external_exasol_db_host,
                external_exasol_db_port=self.external_exasol_db_port,
                external_exasol_bucketfs_port=self.external_exasol_bucketfs_port,
                network_info_dict=network_info_dict,
                attempt=attempt
            )

    def create_wait_for_database_task(self, attempt, database_info_dict, test_container_info_dict):
        return WaitForTestExternalDatabase(
            environment_name=self.environment_name,
            test_container_info_dict=test_container_info_dict,
            database_info_dict=database_info_dict,
            attempt=attempt,
            db_user=self.db_user,
            db_password=self.db_password,
            bucketfs_write_password=self.bucketfs_write_password)
