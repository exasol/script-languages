import os
import unittest

import docker

from exaslct_src.test import utils


class DockerPullTest(unittest.TestCase):

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
            print(f"Finished start container of {registry_container_name}")
            self.test_environment.repository_prefix = f"localhost:{self.registry_port}"
        finally:
            docker_client.close()

    def setUp(self):
        print(f"SetUp {self.__class__.__name__}")
        self.test_environment = utils.ExaslctTestEnvironment(self)
        self.create_registry()
        self.test_environment.clean_images()
        command = f"./exaslct push "
        self.test_environment.run_command(command, track_task_dependencies=False)
        self.test_environment.clean_images()

    def tearDown(self):
        utils.remove_docker_container([self.registry_container.id])
        self.test_environment.close()

    def test_docker_pull(self):
        command = f"./exaslct build "
        self.test_environment.run_command(command, track_task_dependencies=True)
        docker_pull_image_tasks = self.find_all("timers", "DockerPullImageTask", self.test_environment.temp_dir)
        print(docker_pull_image_tasks)
        self.assertEqual(len(docker_pull_image_tasks),3,
                         f"{docker_pull_image_tasks} doesn't contain the expected 3 docker_pull_image_tasks")

    def find_all(self, search_root, search_name, path):
        result = []
        for root, dirs, files in os.walk(path):
            if search_root in root:
                for file in files:
                    if search_name in file:
                        result.append(os.path.join(root, file))
                for dir in dirs:
                    if search_name in dir:
                        result.append(os.path.join(root, dir))
        return result


if __name__ == '__main__':
    unittest.main()
