import ssl
from xmlrpc.client import ServerProxy as xmlrpc

import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.test_runner.database_credentials import DatabaseCredentialsParameter
from exaslct_src.lib.test_runner.external_test_environment_parameter import ExternalDatabaseXMLRPCParameter, \
    ExternalDatabaseHostParameter


class SetupExternalDatabaseHost(DependencyLoggerBaseTask,
                                ExternalDatabaseXMLRPCParameter,
                                ExternalDatabaseHostParameter,
                                DatabaseCredentialsParameter):
    environment_name = luigi.Parameter()
    network_info = JsonPickleParameter(DockerNetworkInfo, significant=False)  # type: DockerNetworkInfo
    attempt = luigi.IntParameter(1)

    def run_task(self):
        database_host = self.external_exasol_db_host
        if self.external_exasol_db_host == "localhost" or \
                self.external_exasol_db_host == "127.0.01":
            database_host = self.network_info.gateway
        self.setup_database()
        database_info = DatabaseInfo(database_host,
                                     self.external_exasol_db_port,
                                     self.external_exasol_bucketfs_port)
        self.return_object(database_info)

    def setup_database(self):
        if self.external_exasol_xmlrpc_host is not None:
            # TODO add option to use unverified ssl
            uri = 'https://{user}:{password}@{host}:{port}/{cluster_name}'.format(
                user=self.external_exasol_xmlrpc_user,
                password=self.external_exasol_xmlrpc_password,
                host=self.external_exasol_xmlrpc_host,
                port=self.external_exasol_xmlrpc_port,
                cluster_name=self.external_exasol_xmlrpc_cluster_name
            )
            server = xmlrpc(uri, context=ssl._create_unverified_context(), verbose=True)
            server.bfsdefault.editBucketFS({'http_port': int(self.external_exasol_bucketfs_port)})
            try:
                server.bfsdefault.addBucket({'bucket_name': 'myudfs', 'public_bucket': True,
                                             'read_password': self.bucketfs_write_password,
                                             'write_password': self.bucketfs_write_password})
            except Exception as e:
                self.logger.info(e)
            try:
                server.bfsdefault.addBucket({'bucket_name': 'jdbc_adapter', 'public_bucket': True,
                                             'read_password': self.bucketfs_write_password,
                                             'write_password': self.bucketfs_write_password})
            except Exception as e:
                self.logger.info(e)
