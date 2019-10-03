import logging
import time
from threading import Thread

from docker.models.containers import Container

from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.test_runner.database_credentials import DatabaseCredentials


class IsDatabaseReadyThread(Thread):

    def __init__(self,
                 logger,
                 database_info: DatabaseInfo,
                 database_credentials: DatabaseCredentials,
                 test_container: Container):
        super().__init__()
        self.logger = logger
        self.database_credentials = database_credentials
        self._database_info = database_info
        self.test_container = test_container
        self.finish = False
        self.is_ready = False
        self.output_db_connection = None
        self.output_bucketfs_connection = None

    def stop(self):
        self.logger.info("Stop IsDatabaseReadyThread")
        self.finish = True

    def run(self):
        db_connection_command = self.create_db_connection_command()
        bucket_fs_connection_command = self.create_bucketfs_connection_command()
        while not self.finish:
            (exit_code_db_connection, self.output_db_connection) = \
                self.test_container.exec_run(cmd=db_connection_command)
            (exit_code_bucketfs_connection, self.output_bucketfs_connection) = \
                self.test_container.exec_run(cmd=bucket_fs_connection_command)
            if exit_code_db_connection == 0 and exit_code_bucketfs_connection == 0:
                self.finish = True
                self.is_ready = True
            time.sleep(1)

    def create_db_connection_command(self):
        username = self.database_credentials.db_user
        password = self.database_credentials.db_password
        connection_options = f"""-c '{self._database_info.host}:{self._database_info.db_port}' -u '{username}' -p '{password}'"""
        cmd = f"""$EXAPLUS {connection_options}  -sql 'select 1;'"""
        bash_cmd = f"""bash -c "{cmd}" """
        return bash_cmd

    def create_bucketfs_connection_command(self):
        username = "w"
        password = self.database_credentials.bucketfs_write_password
        cmd = f"""curl --fail '{username}:{password}@{self._database_info.host}:{self._database_info.bucketfs_port}'"""
        bash_cmd = f"""bash -c "{cmd}" """
        return bash_cmd