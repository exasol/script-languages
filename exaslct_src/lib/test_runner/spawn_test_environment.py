import logging

import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.test_runner.environment_type import EnvironmentType
from exaslct_src.lib.test_runner.spawn_test_environment_parameter import SpawnTestEnvironmentParameter
from exaslct_src.lib.test_runner.spawn_test_environment_with_docker_db import SpawnTestEnvironmentWithDockerDB
from exaslct_src.lib.test_runner.spawn_test_environment_with_external_db import SpawnTestEnvironmentWithExternalDB


class SpawnTestEnvironment(DependencyLoggerBaseTask, SpawnTestEnvironmentParameter):
    DEFAULT_DB_USER = "sys"
    DEFAULT_DATABASE_PASSWORD = "exasol"
    DEFAULT_BUCKETFS_WRITE_PASSWORD = "write"

    logger = logging.getLogger('luigi-interface')
    environment_name = luigi.Parameter()

    def register_required(self):
        task = self._create_spawn_environment_task()
        self._environment_info_future = self.register_dependency(task)

    def _create_spawn_environment_task(self):
        if self.environment_type == EnvironmentType.docker_db:
            return self._create_docker_db_environment()
        else:
            return self._create_external_db_environment()

    def _create_external_db_environment(self):
        if self.external_exasol_db_host is None:
            raise Exception("external_exasol_db_host not set")
        if self.external_exasol_db_port is None:
            raise Exception("external_exasol_db_port not set")
        if self.external_exasol_bucketfs_port is None:
            raise Exception("external_exasol_bucketfs_port not set")
        task = \
            self.create_child_task_with_common_params(
                SpawnTestEnvironmentWithExternalDB,
                db_user=self.external_exasol_db_user,
                db_password=self.external_exasol_db_password,
                bucketfs_write_password=self.external_exasol_bucketfs_write_password
            )
        return task

    def _create_docker_db_environment(self):
        task = \
            self.create_child_task_with_common_params(
                SpawnTestEnvironmentWithDockerDB,
                db_user=self.DEFAULT_DB_USER,
                db_password=self.DEFAULT_DATABASE_PASSWORD,
                bucketfs_write_password=self.DEFAULT_BUCKETFS_WRITE_PASSWORD
            )
        return task

    def run_task(self):
        environment_info = self.get_values_from_future(self._environment_info_future)
        self.return_object(environment_info)
