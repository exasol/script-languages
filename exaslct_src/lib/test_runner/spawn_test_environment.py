import logging

import luigi
from luigi.parameter import ParameterVisibility

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_environment_info_collector import ENVIRONMENT_INFO
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.test_runner.environment_type import EnvironmentType
from exaslct_src.lib.test_runner.spawn_test_environment_parameter import SpawnTestEnvironmentParameter
from exaslct_src.lib.test_runner.spawn_test_environment_with_docker_db import SpawnTestEnvironmentWithDockerDB
from exaslct_src.lib.test_runner.spawn_test_environment_with_external_db import SpawnTestEnvironmentWithExternalDB



class SpawnTestEnvironment(StoppableTask, SpawnTestEnvironmentParameter):
    DEFAULT_DB_USER = "sys"
    DEFAULT_DATABASE_PASSWORD = "exasol"
    DEFAULT_BUCKETFS_WRITE_PASSWORD = "write"

    logger = logging.getLogger('luigi-interface')
    environment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._environment_info_target = luigi.LocalTarget(
            "%s/info/environment/%s/environment_info"
            % (build_config().output_directory,
               self.environment_name))
        if self._environment_info_target.exists():
            self._environment_info_target.remove()

    def requires_tasks(self):
        if self.environment_type == EnvironmentType.docker_db:
            return SpawnTestEnvironmentWithDockerDB(
                environment_name=self.environment_name,
                reuse_database_setup=self.reuse_database_setup,
                reuse_test_container=self.reuse_test_container,
                docker_db_image_name=self.docker_db_image_name,
                docker_db_image_version=self.docker_db_image_version,
                reuse_database=self.reuse_database,
                database_port_forward=self.database_port_forward,
                bucketfs_port_forward=self.bucketfs_port_forward,
                max_start_attempts=self.max_start_attempts,
                db_user=self.DEFAULT_DB_USER,
                db_password=self.DEFAULT_DATABASE_PASSWORD,
                bucketfs_write_password = self.DEFAULT_BUCKETFS_WRITE_PASSWORD
            )
        else:
            if self.external_exasol_db_host is None:
                raise Exception("external_exasol_db_host not set")
            if self.external_exasol_db_port is None:
                raise Exception("external_exasol_db_port not set")
            if self.external_exasol_bucketfs_port is None:
                raise Exception("external_exasol_bucketfs_port not set")
            return SpawnTestEnvironmentWithExternalDB(
                environment_name=self.environment_name,
                reuse_database_setup=self.reuse_database_setup,
                reuse_test_container=self.reuse_test_container,
                external_exasol_db_host=self.external_exasol_db_host,
                external_exasol_db_port = self.external_exasol_db_port,
                external_exasol_bucketfs_port = self.external_exasol_bucketfs_port,
                db_user=self.external_exasol_db_user,
                db_password=self.external_exasol_db_password,
                bucketfs_write_password = self.external_exasol_bucketfs_write_password
            )

    def output(self):
        return {
            ENVIRONMENT_INFO: self._environment_info_target,
        }

    def run_task(self):
        with self.input()[ENVIRONMENT_INFO].open("r") as input_file:
            with self._environment_info_target.open("w") as output_file:
                output_file.write(input_file.read())

