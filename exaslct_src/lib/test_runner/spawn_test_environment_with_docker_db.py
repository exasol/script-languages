import luigi

from exaslct_src.lib.test_runner.prepare_network_for_test_environment import PrepareDockerNetworkForTestEnvironment
from exaslct_src.lib.test_runner.spawn_test_database import SpawnTestDockerDatabase
from exaslct_src.lib.test_runner.spawn_test_environment import SpawnTestEnvironment


class SpawnTestEnvironmentWithDockerDB(SpawnTestEnvironment):

    docker_db_image_name = luigi.OptionalParameter(None)
    docker_db_image_version = luigi.OptionalParameter(None)
    reuse_database = luigi.BoolParameter(False, significant=False)
    database_port_forward = luigi.OptionalParameter(None, significant=False)
    bucketfs_port_forward = luigi.OptionalParameter(None, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.db_container_name = f"""db_container_{self.environment_name}"""

    def create_network_task(self, attempt):
        return PrepareDockerNetworkForTestEnvironment(
                environment_name=self.environment_name,
                test_container_name=self.test_container_name,
                db_container_name=self.db_container_name,
                network_name=self.network_name,
                reuse=self.reuse_database,
                attempt=attempt
            )

    def create_spawn_database_task(self, network_info_dict, attempt):
        return SpawnTestDockerDatabase(
                    environment_name=self.environment_name,
                    db_container_name=self.db_container_name,
                    docker_db_image_version = self.docker_db_image_version,
                    docker_db_image_name = self.docker_db_image_name,
                    network_info_dict=network_info_dict,
                    ip_address_index_in_subnet=0,
                    database_port_forward=self.database_port_forward,
                    bucketfs_port_forward=self.bucketfs_port_forward,
                    reuse_database=self.reuse_database,
                    attempt=attempt
                )