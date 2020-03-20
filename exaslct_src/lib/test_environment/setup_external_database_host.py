import ssl
from time import sleep
from urllib.parse import quote_plus
from xmlrpc.client import ServerProxy

import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.test_environment.data.database_info import DatabaseInfo
from exaslct_src.lib.test_environment.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.test_environment.data.database_credentials import DatabaseCredentialsParameter
from exaslct_src.lib.test_environment.external_test_environment_parameter import ExternalDatabaseXMLRPCParameter, \
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
            cluster = self.getXMLRPCObject()
            self.start_database(cluster)
            cluster.bfsdefault.editBucketFS({'http_port': int(self.external_exasol_bucketfs_port)})
            try:
                cluster.bfsdefault.addBucket({'bucket_name': 'myudfs', 'public_bucket': True,
                                              'read_password': self.bucketfs_write_password,
                                              'write_password': self.bucketfs_write_password})
            except Exception as e:
                self.logger.info(e)
            try:
                cluster.bfsdefault.addBucket({'bucket_name': 'jdbc_adapter', 'public_bucket': True,
                                              'read_password': self.bucketfs_write_password,
                                              'write_password': self.bucketfs_write_password})
            except Exception as e:
                self.logger.info(e)

    def getXMLRPCObject(self, object_name: str = ""):
        uri = 'https://{user}:{password}@{host}:{port}/{cluster_name}/{object_name}'.format(
            user=quote_plus(self.external_exasol_xmlrpc_user),
            password=quote_plus(self.external_exasol_xmlrpc_password),
            host=self.external_exasol_xmlrpc_host,
            port=self.external_exasol_xmlrpc_port,
            cluster_name=self.external_exasol_xmlrpc_cluster_name,
            object_name=object_name
        )
        server = ServerProxy(uri, context=ssl._create_unverified_context())
        return server

    def start_database(self, cluster: ServerProxy):
        storage = self.getXMLRPCObject("storage")
        # wait until all nodes are online
        self.logger.info('Waiting until all nodes are online')
        allNodesOnline = False
        while not allNodesOnline:
            allNodesOnline = True
            for nodeName in cluster.getNodeList():
                nodeState = self.getXMLRPCObject(nodeName).getNodeState()
                if nodeState['status'] != 'Running':
                    allNodesOnline = False
                    break
            sleep(5)
        self.logger.info('All nodes are online now')

        # start EXAStorage
        if not storage.serviceIsOnline():
            if storage.startEXAStorage() != 'OK':
                self.logger.info('EXAStorage has been started successfully')
            else:
                raise Exception('Not able startup EXAStorage!\n')
        elif storage.serviceIsOnline():
            print('EXAStorage already online; continuing startup process')

        # triggering database startup
        for databaseName in cluster.getDatabaseList():
            database = self.getXMLRPCObject('/db_' + quote_plus(databaseName))
            if not database.runningDatabase():
                print('Starting database instance %s' % databaseName)
                database.startDatabase()
            else:
                print('Database instance %s already running' % databaseName)
