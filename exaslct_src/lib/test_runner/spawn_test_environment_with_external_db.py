import logging

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.data.dependency_collector.dependency_database_info_collector import DATABASE_INFO
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.test_runner.prepare_network_for_test_environment import PrepareDockerNetworkForTestEnvironment
from exaslct_src.lib.test_runner.abstract_spawn_test_environment import AbstractSpawnTestEnvironment
from exaslct_src.lib.test_runner.wait_for_external_database import WaitForTestExternalDatabase


class DetermineExternalDatabaseHost(StoppableTask):
    logger = logging.getLogger('luigi-interface')

    environment_name = luigi.Parameter()
    external_exasol_db_host = luigi.Parameter()
    external_exasol_db_port = luigi.Parameter()
    external_exasol_bucketfs_port = luigi.Parameter()
    network_info_dict = luigi.DictParameter(significant=False)
    attempt = luigi.IntParameter(1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._database_info_target = luigi.LocalTarget(
            "%s/info/environment/%s/database/%s/%s/database_info"
            % (build_config().output_directory,
               self.environment_name,
               "external_db",
               self.attempt))
        if self._database_info_target.exists():
            self._database_info_target.remove()

    def output(self):
        return {DATABASE_INFO: self._database_info_target}

    def run_task(self):
        network_info = DockerNetworkInfo.from_dict(self.network_info_dict)
        database_host = self.external_exasol_db_host
        if self.external_exasol_db_host == "localhost" or \
                self.external_exasol_db_host == "127.0.01":
            database_host = network_info.gateway
        database_info = DatabaseInfo(database_host,
                                     self.external_exasol_db_port,
                                     self.external_exasol_bucketfs_port)
        self.write_output(database_info)

    def write_output(self, database_info: DatabaseInfo):
        with self.output()[DATABASE_INFO].open("w") as file:
            file.write(database_info.to_json())


class SpawnTestEnvironmentWithExternalDB(AbstractSpawnTestEnvironment):
    external_exasol_db_host = luigi.OptionalParameter()
    external_exasol_db_port = luigi.Parameter()
    external_exasol_bucketfs_port = luigi.Parameter()

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
        task = yield WaitForTestExternalDatabase(environment_name=self.environment_name,
                                                 test_container_info_dict=test_container_info_dict,
                                                 database_info_dict=database_info_dict,
                                                 attempt=attempt)
        return task
