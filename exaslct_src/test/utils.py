import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path

import docker
import requests


class ExaslctDockerTestEnvironment():
    def __init__(self, name: str, database_host: str,
                 db_username: str, db_password: str,
                 bucketfs_username: str, bucketfs_password: str,
                 database_port: int, bucketfs_port: str):
        self.db_password = db_password
        self.db_username = db_username
        self.database_port = database_port
        self.bucketfs_port = bucketfs_port
        self.bucketfs_username = bucketfs_username
        self.bucketfs_password = bucketfs_password
        self.database_host = database_host
        self.name = name

    def close(self):
        remove_docker_container([f"test_container_{self.name}",
                                 f"db_container_{self.name}"])
        remove_docker_volumes([f"db_container_{self.name}_volume"])

class ExaslctTestEnvironment():

    def __init__(self, test_object):
        self.flavor_path = get_test_flavor()
        self.name = test_object.__class__.__name__
        self._repository_prefix = "exaslct_test"
        if "GOOGLE_CLOUD_BUILD" in os.environ:
            # We need to put the output directories into the workdir, 
            # because only this is shared between the current container and
            # host. Only path within this shared directory can be mounted 
            # to docker container started by exaslct
            temp_dir_prefix_path=Path("./temp_outputs")
            temp_dir_prefix_path.mkdir(exist_ok=True)
            self.temp_dir = tempfile.mkdtemp(dir=temp_dir_prefix_path)
        else:
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

    def spawn_docker_test_environment(self, name) -> ExaslctDockerTestEnvironment:
        parameter = ExaslctDockerTestEnvironment(
            name=self.name + "_" + name,
            database_host="localhost",
            db_username="sys",
            db_password="exasol",
            bucketfs_username="w",
            bucketfs_password="write",
            database_port=find_free_port(),
            bucketfs_port=find_free_port())
        arguments = " ".join([f"--environment-name {parameter.name}",
                              f"--database-port-forward {parameter.database_port}",
                              f"--bucketfs-port-forward {parameter.bucketfs_port}"])
        command = f"./exaslct spawn-test-environment {arguments}"
        self.run_command(command, use_flavor_path=False, use_docker_repository=False)
        if "GOOGLE_CLOUD_BUILD" in os.environ:
            parameter.database_host="178.178.178.178"
            parameter.database_port=8888
            parameter.bucketfs_port=6583
            docker_client = docker.from_env()
            try:
                db_container=docker_client.containers.get("db_container_{parameter.name}")
                cloudbuild_network=docker_client.networks.get("cloudbuild")
                cloudbuild_network.connect(db_container,ipv4_address=parameter.database_hos)
            finally:
                docker_client.close()
        return parameter
    
    def create_registry(self):
        registry_port = utils.find_free_port()
        registry_container_name = self.name.replace("/", "_") + "_registry"
        docker_client = docker.from_env()
        try:
            print("Start pull of registry:2")
            docker_client.images.pull(repository="registry", tag="2")
            print(f"Start container of {registry_container_name}")
            try:
                docker_client.containers.get(registry_container_name).remove(force=True)
            except:
                pass
            registry_container = docker_client.containers.run(
                image="registry:2", name=registry_container_name,
                ports={5000: self.registry_port},
                detach=True
            )
            time.sleep(10)
            print(f"Finished start container of {registry_container_name}")
            if "GOOGLE_CLOUD_BUILD" in os.environ:
                registry_host="178.178.178.179"
                cloudbuild_network=docker_client.networks.get("cloudbuild")
                cloudbuild_network.connect(registry_container,ipv4_address=registry_host)
                self.repository_prefix = f"{registry_host}:5000"
                return registry_container,registry_host,"5000"
            else:
                self.repository_prefix = f"localhost:{registry_port}"
                return registry_container,"localhost",registry_port
        finally:
            docker_client.close()


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

def remove_docker_volumes(volumes):
    docker_client = docker.from_env()
    try:
        for volume in volumes:
            try:
                docker_client.volumes.get(volume).remove(force=True)
            except Exception as e:
                print(e)
    finally:
        docker_client.close()


def get_test_flavor():
    flavor_path = Path(os.path.realpath(__file__)).parent.joinpath("resources/test-flavor")
    return flavor_path



def request_registry_images(registry_host,registry_port, repo_name):
    url = f"http://{registry_host}:{registry_port}/v2/{repo_name}/tags/list"
    result = requests.request("GET", url)
    images = json.loads(result.content.decode("UTF-8"))
    return images


def request_registry_repositories(registry_host,registry_port):
    result = requests.request("GET", f"http://{registry_host}:{registry_port}/v2/_catalog/")
    repositories_ = json.loads(result.content.decode("UTF-8"))["repositories"]
    return repositories_
