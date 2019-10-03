import luigi

from exaslct_src.AbstractMethodException import AbstractMethodException
from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.data.environment_info import EnvironmentInfo
from exaslct_src.lib.test_runner.database_credentials import DatabaseCredentialsParameter
from exaslct_src.lib.test_runner.general_spawn_test_environment_parameter import GeneralSpawnTestEnvironmentParameter
from exaslct_src.lib.test_runner.populate_data import PopulateEngineSmallTestDataToDatabase
from exaslct_src.lib.test_runner.spawn_test_container import SpawnTestContainer
from exaslct_src.lib.test_runner.upload_exa_jdbc import UploadExaJDBC
from exaslct_src.lib.test_runner.upload_virtual_schema_jdbc_adapter import UploadVirtualSchemaJDBCAdapter

DATABASE = "database"

TEST_CONTAINER = "test_container"


class AbstractSpawnTestEnvironment(DependencyLoggerBaseTask,
                                   GeneralSpawnTestEnvironmentParameter,
                                   DatabaseCredentialsParameter):
    environment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_container_name = f"""test_container_{self.environment_name}"""
        self.network_name = f"""db_network_{self.environment_name}"""

    def run_task(self):
        test_environment_info = yield from self._attempt_database_start()
        yield from self._setup_test_database(test_environment_info)
        self.return_object(test_environment_info)

    def _attempt_database_start(self):
        is_database_ready = False
        attempt = 0
        while not is_database_ready and attempt < self.max_start_attempts:
            database_info, is_database_ready, test_container_info = \
                yield from self._start_database(attempt)
            attempt += 1
        if not is_database_ready and not attempt < self.max_start_attempts:
            raise Exception(f"Maximum attempts {attempt} to start the database reached.")
        test_environment_info = \
            EnvironmentInfo(name=self.environment_name,
                            database_info=database_info,
                            test_container_info=test_container_info)
        return test_environment_info

    def _start_database(self, attempt):
        network_info = yield from self._create_network(attempt)
        database_info, test_container_info = \
            yield from self._spawn_database_and_test_container(network_info, attempt)
        is_database_ready = yield from self._wait_for_database(
            database_info, test_container_info, attempt)
        return database_info, is_database_ready, test_container_info

    def _create_network(self, attempt):
        network_info_future = yield from self.run_dependencies(self.create_network_task(attempt))
        network_info = self.get_values_from_future(network_info_future)
        return network_info

    def create_network_task(self, attempt: int):
        raise AbstractMethodException()

    def _spawn_database_and_test_container(self,
                                           network_info: DockerNetworkInfo,
                                           attempt: int):
        database_and_test_container_info_future = \
            yield from self.run_dependencies({
                TEST_CONTAINER: SpawnTestContainer(
                    environment_name=self.environment_name,
                    test_container_name=self.test_container_name,
                    network_info=network_info,
                    ip_address_index_in_subnet=1,
                    reuse_test_container=self.reuse_test_container,
                    attempt=attempt),
                DATABASE: self.create_spawn_database_task(network_info, attempt)
            })
        database_and_test_container_info = \
            self.get_values_from_futures(database_and_test_container_info_future)
        test_container_info = database_and_test_container_info[TEST_CONTAINER]
        database_info = database_and_test_container_info[DATABASE]
        return database_info, test_container_info

    def create_spawn_database_task(self,
                                   network_info: DockerNetworkInfo,
                                   attempt: int):
        raise AbstractMethodException()

    def _wait_for_database(self,
                           database_info: DatabaseInfo,
                           test_container_info: ContainerInfo,
                           attempt: int):
        database_ready_target_future = \
            yield from self.run_dependencies(
                self.create_wait_for_database_task(
                    attempt, database_info, test_container_info))
        is_database_ready = self.get_values_from_futures(database_ready_target_future)
        return is_database_ready

    def create_wait_for_database_task(self,
                                      attempt: int,
                                      database_info: DatabaseInfo,
                                      test_container_info: ContainerInfo):
        raise AbstractMethodException()

    def _setup_test_database(self, test_environment_info: EnvironmentInfo):
        # TODO check if database is setup
        self.logger.info("Setup database")
        upload_tasks = [
            self.create_child_task_with_common_params(
                UploadExaJDBC,
                test_environment_info=test_environment_info,
                reuse_uploaded=self.reuse_database_setup),
            self.create_child_task_with_common_params(
                UploadVirtualSchemaJDBCAdapter,
                test_environment_info=test_environment_info,
                reuse_uploaded=self.reuse_database_setup),
            self.create_child_task_with_common_params(
                PopulateEngineSmallTestDataToDatabase,
                test_environment_info=test_environment_info,
                reuse_data=self.reuse_database_setup
            )]
        yield from self.run_dependencies(upload_tasks)
