import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from exaslct_src.test import utils


class DockerLoadTest(unittest.TestCase):

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.TestEnvironment(self)
        self.save_path = self.test_environment.temp_dir + "/save_dir"
        self.test_environment.clean_images()
        self.save()
        self.test_environment.clean_images()

    def save(self):
        command = f"./exaslct save --save-directory {self.save_path} "
        self.test_environment.run_command(command, track_task_dependencies=False)
        for path, subdirs, files in os.walk(self.save_path):
            for x in files:
                print(Path(path).joinpath(x))

    def run_command(self, command: str):
        completed_process = subprocess.run(shlex.split(command))
        completed_process.check_returncode()

    def tearDown(self):
        try:
            self.test_environment.close()
        except Exception as e:
            print(e)

    def test_docker_load(self):
        command = f"./exaslct build --cache-directory {self.save_path} "
        self.test_environment.run_command(command, track_task_dependencies=True)



if __name__ == '__main__':
    unittest.main()
