import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from exaslct_src.test import utils


class DockerExportTest(unittest.TestCase):
    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.TestEnvironment(self)
        self.export_path = self.test_environment.temp_dir + "/export_dir"
        self.test_environment.clean_images()

    def tearDown(self):
        try:
            self.test_environment.close()
        except Exception as e:
            print(e)

    def test_docker_export(self):
        command=f"./exaslct export --export-path {self.export_path}"
        self.test_environment.run_command(command,track_task_dependencies=True)
        exported_files = os.listdir(self.export_path)
        self.assertTrue(len(exported_files) == 1,
                        f"Did not found saved files for repository {self.test_environment.repository_name} "
                        f"in list {exported_files}")


if __name__ == '__main__':
    unittest.main()
