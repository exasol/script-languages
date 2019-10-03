import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.test_runner.external_test_environment_parameter import ExternalDatabaseHostParameter


class DetermineExternalDatabaseHost(DependencyLoggerBaseTask,ExternalDatabaseHostParameter):

    environment_name = luigi.Parameter()
    network_info = JsonPickleParameter(DockerNetworkInfo,significant=False) # type: DockerNetworkInfo
    attempt = luigi.IntParameter(1)

    def run_task(self):
        database_host = self.external_exasol_db_host
        if self.external_exasol_db_host == "localhost" or \
                self.external_exasol_db_host == "127.0.01":
            database_host = self.network_info.gateway
        database_info = DatabaseInfo(database_host,
                                     self.external_exasol_db_port,
                                     self.external_exasol_bucketfs_port)
        self.return_object(database_info)
