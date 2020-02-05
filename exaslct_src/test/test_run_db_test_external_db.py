import unittest
from subprocess import CalledProcessError

from exaslct_src.test import utils


class DockerRunDBTestExternalDBTest(unittest.TestCase):

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.ExaslctTestEnvironment(self)
        self.test_environment.clean_images()
        self.docker_environment = self.test_environment.spawn_docker_test_environment("test")

    def tearDown(self):
        try:
            self.docker_environment.close()
        except Exception as e:
            print(e)
        try:
            self.test_environment.close()
        except Exception as e:
            print(e)

    def test_run_db_tests_external_db(self):
        arguments = " ".join([
            f"--environment-type external_db",
            f"--external-exasol-db-host localhost", # localhost gets translated in exaslct to the Gateway adress of the docker environment network, because thats typically the IP Adress of the bridge to the host, for google cloud this means it should be able to connect to the db via the port forwards from the test container 
            f"--external-exasol-db-port 8888",
            f"--external-exasol-bucketfs-port 6666",
            f"--external-exasol-db-user {self.docker_environment.db_username}",
            f"--external-exasol-db-password {self.docker_environment.db_password}",
            f"--external-exasol-bucketfs-write-password {self.docker_environment.bucketfs_password}",
        ])
        command = f"./exaslct run-db-test {arguments}"
        self.test_environment.run_command(
                              command, track_task_dependencies=True)


if __name__ == '__main__':
    unittest.main()
