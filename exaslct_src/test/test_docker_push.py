import time
import unittest

import docker

from exaslct_src.test import utils


class DockerPushTest(unittest.TestCase):

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.ExaslctTestEnvironment(self)
        self.registry_container,self.registry_host,self.registry_port=self.test_environment.create_registry()
        print("registry:", utils.request_registry_repositories(self.registry_host,self.registry_port))
        self.test_environment.clean_images()

    def tearDown(self):
        utils.remove_docker_container([self.registry_container.id])
        self.test_environment.close()

    def test_docker_push(self):
        command = f"./exaslct push "
        self.test_environment.run_command(command, track_task_dependencies=True)
        print("repos:", utils.request_registry_repositories(self.registry_host,self.registry_port))
        images = utils.request_registry_images(self.registry_host,self.registry_port, "dockerpushtest")
        print("images", images)
        self.assertEqual(len(images["tags"]),10, f"{images} doesn't have the expected 10 tags, it only has {len(images['tags'])}")


if __name__ == '__main__':
    unittest.main()
