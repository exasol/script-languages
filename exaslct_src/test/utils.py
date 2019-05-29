import os
import shlex
import shutil
import socket
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path

import docker


class TestEnvironment():

    def __init__(self, test_object):
        self.flavor_path = get_test_flavor()
        self.name = test_object.__class__.__name__
        self._repository_prefix="exaslct_test"
        self.temp_dir = tempfile.mkdtemp()
        self._update_attributes()

    @property
    def repository_prefix(self):
        return self._repository_prefix

    @repository_prefix.setter
    def repository_prefix(self, value):
        self._repository_prefix = value
        self._update_attributes()

    def _update_attributes(self):
        self.repository_name = f"{self._repository_prefix.lower()}/{self.name.lower()}"  # docker repository names must be lowercase
        self.flavor_path_argument = f"--flavor-path {get_test_flavor()}"
        self.docker_repository_arguments = f"--source-docker-repository-name {self.repository_name} --target-docker-repository-name {self.repository_name}"
        self.clean_docker_repository_arguments = f"--docker-repository-name {self.repository_name}"
        self.output_directory_arguments = f"--output-directory {self.temp_dir}"
        self.task_dependencies_argument = " ".join([f"--task-dependencies-dot-file {self.name}.dot", ])

    def clean_images(self):
        self.run_command(f"./exaslct clean-flavor-images", clean=True)

    def run_command(self, command: str,
                    use_flavor_path: bool = True,
                    use_docker_repository: bool = True,
                    track_task_dependencies: bool = False,
                    clean: bool = False):
        command = f"{command} {self.output_directory_arguments}"
        if track_task_dependencies:
            command = f"{command} {self.task_dependencies_argument}"
        if use_flavor_path:
            command = f"{command} {self.flavor_path_argument}"
        if use_docker_repository and not clean:
            command = f"{command} {self.docker_repository_arguments}"
        if use_docker_repository and clean:
            command = f"{command} {self.clean_docker_repository_arguments}"
        print()
        print(f"command: {command}")
        completed_process = subprocess.run(shlex.split(command))
        completed_process.check_returncode()

    def close(self):
        try:
            self.clean_images()
        except Exception as e:
            print(e)
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(e)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', port))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return port


def remove_docker_container(containers):
    docker_client = docker.from_env()
    try:
        for container in containers:
            try:
                docker_client.containers.get(container).remove(force=True)
            except Exception as e:
                print(e)
    finally:
        docker_client.close()


def get_test_flavor():
    flavor_path = Path(os.path.realpath(__file__)).parent.joinpath("resources/test-flavor")
    return flavor_path
