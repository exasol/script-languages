import os
import unittest

from exasol_integration_test_docker_environment.test import utils



class DockerExportTest(unittest.TestCase):
    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.ExaslctTestEnvironment(self)
        self.export_path = self.test_environment.temp_dir + "/export_dir"
        self.test_environment.clean_images()

    def tearDown(self):
        try:
            self.test_environment.close()
        except Exception as e:
            print(e)

    def test_docker_export(self):
        command = f"./exaslct export --export-path {self.export_path}"
        self.test_environment.run_command(command, track_task_dependencies=True)
        exported_files = os.listdir(self.export_path)
        self.assertEqual(sorted(list(exported_files)), sorted(['test-flavor_release.tar.gz', 'test-flavor_release.tar.gz.sha512sum']),
                         f"Did not found saved files for repository {self.test_environment.repository_name} "
                         f"in list {exported_files}")


if __name__ == '__main__':
    unittest.main()
