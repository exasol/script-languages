import unittest

from exaslct_src.test_environment.src.test import utils


class RunDBTestDockerPassThroughTest(unittest.TestCase):

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment=utils.ExaslctTestEnvironment(self)
        self.test_environment.clean_images()

    def tearDown(self):
        try:
            self.test_environment.close()
        except Exception as e:
            print(e)

    def test_docker_test_environment(self):
        command = f"./exaslct run-db-test --test-file docker_environment_test.py"
        self.test_environment.run_command(command, track_task_dependencies=True)


if __name__ == '__main__':
    unittest.main()
