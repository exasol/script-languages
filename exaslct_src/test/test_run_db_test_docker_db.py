import unittest
from subprocess import CalledProcessError

from exaslct_src.test import utils


class DockerRunDBTestDockerDBTest(unittest.TestCase):

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.ExaslctTestEnvironment(self)
        self.test_environment.clean_images()

    def tearDown(self):
        self.remove_docker_container()
        self.test_environment.close()

    def remove_docker_container(self):
        utils.remove_docker_container([f"test_container_{self.test_environment.name}",
                                       f"db_container_{self.test_environment.name}"])

    def test_run_db_tests_docker_db(self):
        command = f"./exaslct run-db-test "
        self.assertRaises(CalledProcessError,
                          lambda: self.test_environment.run_command(
                              command, track_task_dependencies=True))


if __name__ == '__main__':
    unittest.main()
