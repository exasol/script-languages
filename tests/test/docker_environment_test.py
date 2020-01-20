#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../lib'))

import udf

def is_docker_environment():
    return os.environ["TEST_ENVIRONMENT_TYPE"]=="docker_db"

def db_container_name_is_set():
    return os.environ["TEST_DB_CONTAINER_NAME"] is not None

def test_network_name_is_set():
    return os.environ["TEST_NETWORK_NAME"] is not None

class DockerInTestEnvironment(udf.TestCase):

    @udf.skipIfNot(udf.docker_available, reason="This test requires a docker daemon to start 3rd party software in docker container")
    def test_docker(self):
        import docker
        client=udf.get_docker_client()
        print(client.containers.list())

    @udf.skipIf(is_docker_environment, reason="This test requires a environment with a docker-db")
    def test_docker_environment(self):
        self.fail()

    @udf.skipIfNot(udf.docker_available, reason="This test requires a docker daemon to start 3rd party software in docker container")
    @udf.skipIfNot(test_network_name_is_set, reason="This test requires a test network to be set")
    def test_network_name_is_set(self):
        import docker
        client=udf.get_docker_client()
        print(client.networks.get(os.environ["TEST_NETWORK_NAME"]))

    @udf.skipIfNot(udf.docker_available, reason="This test requires a docker daemon to start 3rd party software in docker container")
    @udf.skipIfNot(db_container_name_is_set, reason="This test requires the db container name to be set")
    def test_db_container_name_is_set(self):
        import docker
        client=udf.get_docker_client()
        print(client.containers.get(os.environ["TEST_DB_CONTAINER_NAME"]))

    @udf.skipIfNot(udf.docker_available, reason="This test requires a docker daemon to start 3rd party software in docker container")
    @udf.skipIfNot(test_network_name_is_set, reason="This test requires a test network to be set")
    @udf.skipIfNot(is_docker_environment, reason="This test requires a environment with a docker-db")
    def test_connect_from_udf_to_other_container(self):
        import docker
        client=udf.get_docker_client()
        container=client.containers.run(image="busybox:1",command="nc -v -l -s 0.0.0.0 -p 7777",detach=True,network=os.environ["TEST_NETWORK_NAME"])
        try:
            schema="docker_envrionment_test"
            self.query(udf.fixindent("DROP SCHEMA %s CASCADE"%schema),ignore_errors=True)
            self.query(udf.fixindent("CREATE SCHEMA %s"%schema))
            self.query(udf.fixindent("OPEN SCHEMA %s"%schema))
            self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON SCALAR SCRIPT connect_container(a int)  returns int AS
                import socket
                def run(ctx):
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(("%s", 7777))
                    return 0
                /
                ''' % (container.attrs['NetworkSettings']['IPAddress'])))
            print(container.attrs['NetworkSettings']['IPAddress'])
            self.query("select connect_container(1)")
            print(container.logs())
        finally:
            try:
                self.query(udf.fixindent("DROP SCHEMA %s CASCADE"%schema))
            except:
                pass
            try:
                container.kill()
            except:
                pass
if __name__ == '__main__':
    udf.main()

