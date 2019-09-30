import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from exaslct_src.test import utils


class DockerSaveTest(unittest.TestCase):

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.ExaslctTestEnvironment(self)
        self.save_path = self.test_environment.temp_dir + "/save_dir"
        self.test_environment.clean_images()

    def run_command(self, command: str):
        completed_process = subprocess.run(shlex.split(command))
        completed_process.check_returncode()

    def tearDown(self):
        try:
            self.test_environment.close()
        except Exception as e:
            print(e)

    def test_docker_save(self):
        command = f"./exaslct save --save-directory {self.save_path} "
        self.test_environment.run_command(command,track_task_dependencies=True)
        saved_files = os.listdir(Path(self.save_path).joinpath(self.test_environment.repository_name).parent)
        self.assertTrue(len(saved_files) > 0,
                        f"Did not found saved files for repository {self.test_environment.repository_name} "
                        f"in list {saved_files}")


if __name__ == '__main__':
    unittest.main()
