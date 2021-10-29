#!/usr/bin/env python3

import time

from exasol_python_test_framework import udf
from exasol_python_test_framework import docker_db_environment


class PysftpConnectionTest(udf.TestCase):

    @udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
    def test_pysftp_connect_python3(self):
        self.pysftp_connect("PYTHON3")

    @udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
    def test_pysftp_connect_python2(self):
        self.pysftp_connect("PYTHON")

    def pysftp_connect(self, python_version):
        schema = "test_pysftp_connect" + python_version
        env = docker_db_environment.DockerDBEnvironment(schema)
        try:
            self.query(udf.fixindent("DROP SCHEMA %s CASCADE" % schema), ignore_errors=True)
            self.query(udf.fixindent("CREATE SCHEMA %s" % schema))
            self.query(udf.fixindent("OPEN SCHEMA %s" % schema))
            self.query(udf.fixindent('''
                CREATE OR REPLACE {python_version} SCALAR SCRIPT connect_container(host varchar(1000), port int,username varchar(1000),password varchar(1000), input_string varchar(1000))  returns VARCHAR(100000) AS
                import socket
                import io
                import traceback
                import sys
                def run(ctx):
                    import pysftp
                    cnopts = pysftp.CnOpts()
                    cnopts.hostkeys = None 
                    try:
                        with pysftp.Connection(ctx.host, username=ctx.username, password=ctx.password,cnopts=cnopts) as sftp:
                            with sftp.cd("tmp"):
                                input_buffer = io.StringIO(ctx.input_string)
                                sftp.putfo(input_buffer,"test_file")
                                output_buffer = io.BytesIO()
                                written=sftp.getfo('test_file',output_buffer)
                                value=output_buffer.getvalue()
                                value_decoded=value.decode("utf-8")
                                return value_decoded
                    except:
                        return traceback.format_exc()
                        
                /
                '''.format(python_version=python_version)))
            env.get_client().images.pull("panubo/sshd", tag="1.1.0")
            container = env.run(name="sshd_sftp", image="panubo/sshd:1.1.0",
                                environment=["SSH_USERS=test_user:1000:1000",
                                             "SSH_ENABLE_PASSWORD_AUTH=true", "SFTP_MODE=true"],
                                tmpfs={'/data': 'size=1M,uid=0'})
            print(container.logs())
            time.sleep(10)
            print(container.logs())
            result = container.exec_run(cmd=''' sh -c "echo 'test_user:test_user' | chpasswd" ''')
            result = container.exec_run(cmd='''mkdir /data/tmp''')
            result = container.exec_run(cmd='''chmod 777 /data/tmp''')
            time.sleep(5)
            print(result)
            host = env.get_ip_address_of_container(container)
            rows = self.query("select connect_container('%s',%s,'test_user','test_user','success')" % (host, 22))
            self.assertRowsEqual([("success",)], rows)
            print(container.logs())
        finally:
            try:
                self.query(udf.fixindent("DROP SCHEMA %s CASCADE" % schema))
            except:
                pass
            try:
                env.close()
            except:
                pass


if __name__ == '__main__':
    udf.main()
