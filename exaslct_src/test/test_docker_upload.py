import unittest

from exaslct_src.test import utils

class DockerUploadTest(unittest.TestCase):

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment=utils.ExaslctTestEnvironment(self)
        self.test_environment.clean_images()
        self.docker_environment = self.test_environment.spawn_docker_test_environment("DockerUploadTest")

    def tearDown(self):
        try:
            self.docker_environment.close()
        except Exception as e:
            print(e)
        try:
            self.test_environment.close()
        except Exception as e:
            print(e)

    def test_docker_upload(self):
        self.path_in_bucket = "test"
        self.release_name = "TEST"
        self.bucketfs_name = "bfsdefault"
        self.bucket_name = "default"
        arguments = " ".join([
                              f"--database-host {self.docker_environment.database_host}",
                              f"--bucketfs-port {self.docker_environment.bucketfs_port}",
                              f"--bucketfs-username {self.docker_environment.bucketfs_username}",
                              f"--bucketfs-password {self.docker_environment.bucketfs_password}",
                              f"--bucketfs-name {self.bucketfs_name}",
                              f"--bucket-name {self.bucket_name}",
                              f"--path-in-bucket {self.path_in_bucket}",
                              f"--no-bucketfs-https",
                              f"--release-name {self.release_name}",
                              ])
        command = f"./exaslct upload {arguments}"

        self.test_environment.run_command(command,track_task_dependencies=True)


if __name__ == '__main__':
    unittest.main()
