#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
import docker_db_environment

class ParamikoConnectionTest(udf.TestCase):

    @udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
    def test_connect_via_paramiko_to_ssh(self):
        schema="test_connect_via_paramiko_to_ssh"
        env=docker_db_environment.DockerDBEnvironment(schema)
        try:
            self.query(udf.fixindent("DROP SCHEMA %s CASCADE"%schema),ignore_errors=True)
            self.query(udf.fixindent("CREATE SCHEMA %s"%schema))
            self.query(udf.fixindent("OPEN SCHEMA %s"%schema))
            self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON SCALAR SCRIPT connect_container(host varchar(1000), port int,username varchar(1000),password varchar(1000))  returns int AS
                import socket
                def run(ctx):
                    import paramiko

                    ssh = paramiko.SSHClient()
                    ssh.load_system_host_keys()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(hostname=self.hostname,
                                     port = self.port,
                                     username=self.username,
                                     password=self.password)
                    ssh.close()

                    return 0
                /
                '''))
            container=env.run(name="sftp",image="writl/sftp:latest")
            host=env.get_ip_address_of_container(container)
            self.query("select connect_container('%s',%s,'sftp','c83eDteUDT')"%(host,22))
            self.assertTrue("connect" in container.logs())
        finally:
            try:
                self.query(udf.fixindent("DROP SCHEMA %s CASCADE"%schema))
            except:
                pass
            try:
                env.close()
            except:
                pass


if __name__ == '__main__':
    udf.main()

