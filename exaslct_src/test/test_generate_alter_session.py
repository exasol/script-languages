import unittest

import docker

from exaslct_src.lib.utils.docker_utils import find_images_by_tag
from exaslct_src.test import utils


class GenerateAlterSessionTest(unittest.TestCase):

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.ExaslctTestEnvironment(self)

    def test_generate_alter_session_with_path_in_bucket(self):
        command = f"./exaslct generate-alter-session --bucketfs-name bfsdefault --bucket-name default --path-in-bucket path --container-name container"
        completed_process=self.test_environment.run_command(command,use_docker_repository=False, use_output_directory=False, capture_output=True)
        self.assertIn("ALTER SESSION SET SCRIPT_LANGUAGES='PYTHON=localzmq+protobuf:///bfsdefault/default/path/container?lang=python#buckets/bfsdefault/default/path/container/exaudf/exaudfclient';",completed_process.stdout.decode("UTF-8"))


    def test_generate_alter_session_without_path_in_bucket(self):
        command = f"./exaslct generate-alter-session --bucketfs-name bfsdefault --bucket-name default --container-name container"
        completed_process=self.test_environment.run_command(command,use_docker_repository=False, use_output_directory=False, capture_output=True)
        self.assertIn("ALTER SESSION SET SCRIPT_LANGUAGES='PYTHON=localzmq+protobuf:///bfsdefault/default/container?lang=python#buckets/bfsdefault/default/container/exaudf/exaudfclient';",completed_process.stdout.decode("UTF-8"))

if __name__ == '__main__':
    unittest.main()
