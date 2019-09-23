import logging

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_environment_info_collector import ENVIRONMENT_INFO
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.test_runner.environment_type import EnvironmentType
from exaslct_src.lib.test_runner.spawn_test_environment_with_docker_db import SpawnTestEnvironmentWithDockerDB
from exaslct_src.lib.test_runner.spawn_test_environment_with_external_db import SpawnTestEnvironmentWithExternalDB


class SpawnTestEnvironmentParameter():
    environment_type = luigi.EnumParameter(enum=EnvironmentType)

    reuse_database_setup = luigi.BoolParameter(False, significant=False)
    reuse_test_container = luigi.BoolParameter(False, significant=False)

    external_exasol_db_host = luigi.OptionalParameter()
    external_exasol_db_port = luigi.OptionalParameter()
    external_exasol_bucketfs_port = luigi.Parameter()

    docker_db_image_name = luigi.OptionalParameter(None)
    docker_db_image_version = luigi.OptionalParameter(None)
    reuse_database = luigi.BoolParameter(False, significant=False)
    database_port_forward = luigi.OptionalParameter(None, significant=False)
    bucketfs_port_forward = luigi.OptionalParameter(None, significant=False)

    max_start_attempts = luigi.IntParameter(2, significant=False)


class SpawnTestEnvironment(StoppableTask, SpawnTestEnvironmentParameter):
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
                max_start_attempts=self.max_start_attempts
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
                external_exasol_bucketfs_port = self.external_exasol_bucketfs_port
            )

    def output(self):
        return {
            ENVIRONMENT_INFO: self._environment_info_target,
        }

    def run_task(self):
        with self.input()[ENVIRONMENT_INFO].open("r") as input_file:
            with self._environment_info_target.open("w") as output_file:
                output_file.write(input_file.read())

