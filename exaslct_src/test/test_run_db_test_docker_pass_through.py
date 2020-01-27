import unittest
from subprocess import CalledProcessError

from exaslct_src.test import utils
import docker

class DockerTestEnvironmentTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment=utils.ExaslctTestEnvironment(self)
        self.test_environment.clean_images()
        self.docker__environment_name = self.__class__.__name__
        self.docker_environment = self.test_environment.spawn_docker_test_environment(self.docker__environment_name)

    @classmethod
    def tearDownClass(self):
        try:
            self.docker_environment.close()
        except Exception as e:
            print(e)
        try:
            self.test_environment.close()
        except Exception as e:
            print(e)

    def test_docker_test_environment(self):
        command = f"./exaslct run-db-test --test-file docker_environment_test.py"
        self.test_environment.run_command(command, track_task_dependencies=True)


if __name__ == '__main__':
    unittest.main()
