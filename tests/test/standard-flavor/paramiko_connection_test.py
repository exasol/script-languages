#!/usr/bin/env python2.7

import os
import sys
import time 

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
import docker_db_environment

class ParamikoConnectionTest(udf.TestCase):

    @udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
    def test_connect_via_paramiko_to_ssh_python3(self):
        self.connect_via_paramiko_to_ssh("PYTHON3")


    @udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
    def test_connect_via_paramiko_to_ssh_python2(self):
        self.connect_via_paramiko_to_ssh("PYTHON")


    def connect_via_paramiko_to_ssh(self, python_version):
        schema="test_connect_via_paramiko_to_ssh_"+python_version
        env=docker_db_environment.DockerDBEnvironment(schema)
        try:
            self.query(udf.fixindent("DROP SCHEMA %s CASCADE"%schema),ignore_errors=True)
            self.query(udf.fixindent("CREATE SCHEMA %s"%schema))
            self.query(udf.fixindent("OPEN SCHEMA %s"%schema))
            self.query(udf.fixindent('''
                CREATE OR REPLACE {python_version} SCALAR SCRIPT connect_container(host varchar(1000), port int,username varchar(1000),password varchar(1000))  returns VARCHAR(100000) AS
                import socket
                def run(ctx):
                    import paramiko
                    try:
                        ssh = paramiko.SSHClient()
                        ssh.load_system_host_keys()
                        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                        ssh.connect(hostname=ctx.host,port = ctx.port,username=ctx.username,password=ctx.password)
                        ssh.close()
                        return "success"
                    except paramiko.ssh_exception.NoValidConnectionsError as e:
                        return "%s,%s,%s,%s,%s"%(str(e.errors),ctx.host,ctx.port,ctx.username,ctx.password)
                /
                '''.format(python_version=python_version)))
            container=env.run(name="ssshd",image="panubo/sshd",environment=["SSH_USERS=test_user:1000:1000","SSH_ENABLE_PASSWORD_AUTH=true"])
            time.sleep(10)
            result=container.exec_run(cmd=''' sh -c "echo 'test_user:test_user' | chpasswd" ''')
            time.sleep(5)
            print(result)
            host=env.get_ip_address_of_container(container)
            rows=self.query("select connect_container('%s',%s,'test_user','test_user')"%(host,22))
            self.assertRowsEqual([("success",)], rows)
            print(container.logs())
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

