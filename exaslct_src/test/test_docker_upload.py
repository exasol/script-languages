import unittest

from exaslct_src.test import utils

class DockerUploadTest(unittest.TestCase):

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment=utils.TestEnvironment(self)
        self.remove_docker_container()
        self.test_environment.clean_images()
        self.spawn_docker_test_environment()

    def spawn_docker_test_environment(self):
        self.database_host = "localhost"
        self.bucketfs_username = "w"
        self.bucketfs_password = "write"
        self.bucketfs_name = "bfsdefault"
        self.bucket_name = "default"
        self.database_port = utils.find_free_port()
        self.bucketfs_port = utils.find_free_port()
        arguments = " ".join([f"--environment-name {self.test_environment.name}",
                              f"--database-port-forward {self.database_port}",
                              f"--bucketfs-port-forward {self.bucketfs_port}"])
        command = f"./exaslct spawn-test-environment {arguments}"
        self.test_environment.run_command(command,use_flavor_path=False,use_docker_repository=False)

    def tearDown(self):
        try:
            self.remove_docker_container()
        except Exception as e:
            print(e)
        try:
            self.test_environment.close()
        except Exception as e:
            print(e)

    def remove_docker_container(self):
        utils.remove_docker_container([f"test_container_{self.test_environment.name}",
                                       f"db_container_{self.test_environment.name}"])

    def test_docker_upload(self):
        self.path_in_bucket = "test"
        self.release_name = "TEST"
        arguments = " ".join([
                              f"--database-host {self.database_host}",
                              f"--bucketfs-port {self.bucketfs_port}",
                              f"--bucketfs-username {self.bucketfs_username}",
                              f"--bucketfs-password {self.bucketfs_password}",
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
