import time
import unittest

import docker

from exaslct_src.test import utils


class DockerPushTest(unittest.TestCase):

    def create_registry(self):
        self.registry_port = utils.find_free_port()
        registry_container_name = self.test_environment.name.replace("/", "_") + "_registry"
        docker_client = docker.from_env()
        try:
            print("Start pull of registry:2")
            docker_client.images.pull(repository="registry", tag="2")
            print(f"Start container of {registry_container_name}")
            try:
                docker_client.containers.get(registry_container_name).remove(force=True)
            except:
                pass
            self.registry_container = docker_client.containers.run(
                image="registry:2", name=registry_container_name,
                ports={5000: self.registry_port},
                detach=True
            )
            time.sleep(10)
            print(f"Finished start container of {registry_container_name}")
            self.test_environment.repository_prefix = f"localhost:{self.registry_port}"
        finally:
            docker_client.close()

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.ExaslctTestEnvironment(self)
        self.create_registry()
        print("registry:", utils.request_registry_repositories(self.registry_port))
        self.test_environment.clean_images()

    def tearDown(self):
        utils.remove_docker_container([self.registry_container.id])
        self.test_environment.close()

    def test_docker_push(self):
        command = f"./exaslct push "
        self.test_environment.run_command(command, track_task_dependencies=True)
        print("repos:", utils.request_registry_repositories(self.registry_port))
        images = utils.request_registry_images(self.registry_port, "dockerpushtest")
        print("images", images)
        self.assertEqual(len(images["tags"]),10, f"{images} doesn't have the expected 10 tags, it only has {len(images['tags'])}")


if __name__ == '__main__':
    unittest.main()
