#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../lib'))

import udf

class DockerInTestEnvironment(udf.TestCase):

    @udf.skipIfNot(udf.docker_available, reason="This test requires a docker daemon to start 3rd party software in docker container")
    def test_docker(self):
        import docker
        client=udf.get_docker_client()
        print(client.containers.list())

if __name__ == '__main__':
    udf.main()

