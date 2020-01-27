#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../lib'))

import udf
import docker_db_environment


class DockerDBEnvironmentTest(udf.TestCase):

    @udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
    def test_connect_from_udf_to_other_container(self):
        schema="test_connect_from_udf_to_other_container"
        env=docker_db_environment.DockerDBEnvironment(schema)
        try:
            self.query(udf.fixindent("DROP SCHEMA %s CASCADE"%schema),ignore_errors=True)
            self.query(udf.fixindent("CREATE SCHEMA %s"%schema))
            self.query(udf.fixindent("OPEN SCHEMA %s"%schema))
            self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON SCALAR SCRIPT connect_container(host varchar(1000), port int)  returns int AS
                import socket
                def run(ctx):
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((ctx.host, ctx.port))
                    return 0
                /
                '''))
            container=env.run(name="netcat",image="busybox:1",command="nc -v -l -s 0.0.0.0 -p 7777",)
            host=env.get_ip_address_of_container(container)
            self.query("select connect_container('%s',%s)"%(host,7777))
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

