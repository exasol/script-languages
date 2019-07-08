import pathlib
from typing import Dict, Any

import jsonpickle
import luigi
import netaddr
from luigi import LocalTarget

from exaslct_src.lib.analyze_test_container import AnalyzeTestContainer, DockerTestContainerBuild
from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.dependency_collector.dependency_container_info_collector import CONTAINER_INFO
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.test_runner.create_export_directory import CreateExportDirectory
from exaslct_src.lib.stoppable_task import StoppableTask


class SpawnTestContainer(StoppableTask):
    environment_name = luigi.Parameter()
    test_container_name = luigi.Parameter()
    network_info_dict = luigi.DictParameter(significant=False)
    ip_address_index_in_subnet = luigi.IntParameter(significant=False)
    attempt = luigi.IntParameter(1)
    reuse_test_container = luigi.BoolParameter(False, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._client = docker_client_config().get_client()

        if self.ip_address_index_in_subnet < 0:
            raise Exception(
                "ip_address_index_in_subnet needs to be greater than 0 got %s"
                % self.ip_address_index_in_subnet)
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._test_container_info_target = luigi.LocalTarget(
            "%s/info/environment/%s/test-container/%s/%s/container_info"
            % (build_config().output_directory,
               self.environment_name,
               self.test_container_name,
               self.attempt))
        if self._test_container_info_target.exists():
            self._test_container_info_target.remove()

    def output(self):
        return {CONTAINER_INFO: self._test_container_info_target}

    def requires_tasks(self):
        return {"test_container_image": DockerTestContainerBuild(),
                "export_directory": CreateExportDirectory()}

    def run_task(self):
        network_info = DockerNetworkInfo.from_dict(self.network_info_dict)
        subnet = netaddr.IPNetwork(network_info.subnet)
        ip_address = str(subnet[2 + self.ip_address_index_in_subnet])
        container_info = None
        if network_info.reused and self.reuse_test_container:
            container_info = self.try_to_reuse_test_container(ip_address, network_info)
        if container_info is None:
            container_info = self.create_test_container(ip_address, network_info)
        with self.output()[CONTAINER_INFO].open("w") as file:
            file.write(container_info.to_json())

    def try_to_reuse_test_container(self, ip_address: str, network_info: DockerNetworkInfo) -> ContainerInfo:
        self.logger.info("Task %s: Try to reuse test container %s",
                         self.__repr__(), self.test_container_name)
        container_info = None
        try:
            container_info = self.get_container_info(ip_address, network_info)
        except Exception as e:
            self.logger.warning("Task %s: Tried to reuse test container %s, but got Exeception %s. "
                                "Fallback to create new database.", self.__repr__(), self.test_container_name, e)
        return container_info

    def create_test_container(self, ip_address, network_info) -> ContainerInfo:
        test_container_image_info = self.get_test_container_image_info(self.input())
        # A later task which uses the test_container needs the exported container,
        # but to access exported container from inside the test_container,
        # we need to mount the release directory into the test_container.
        exports_host_path = pathlib.Path(self.get_release_directory()).absolute()
        tests_host_path = pathlib.Path("./tests").absolute()
        test_container = \
            self._client.containers.create(
                image=test_container_image_info.get_target_complete_name(),
                name=self.test_container_name,
                network_mode=None,
                command="sleep infinity",
                detach=True,
                volumes={
                    exports_host_path: {
                        "bind": "/exports",
                        "mode": "ro"
                    },
                    tests_host_path: {
                        "bind": "/tests_src",
                        "mode": "ro"
                    }
                })
        self._client.networks.get(network_info.network_name).connect(test_container, ipv4_address=ip_address)
        test_container.start()
        test_container.exec_run(cmd="cp -r /tests_src /tests")
        container_info = self.get_container_info(ip_address, network_info)
        return container_info

    def get_container_info(self, ip_address, network_info:DockerNetworkInfo)->ContainerInfo:
        test_container = self._client.containers.get(self.test_container_name)
        if test_container.status != "running":
            raise Exception(f"Container {self.test_container_name} not running")
        container_info = ContainerInfo(container_name=self.test_container_name,
                                       ip_address=ip_address,
                                       network_info=network_info)
        return container_info

    def get_release_directory(self):
        return pathlib.Path(self.input()["export_directory"].path).absolute().parent

    def get_test_container_image_info(self, input: Dict[str, LocalTarget]) -> ImageInfo:
        with input["test_container_image"].open("r") as f:
            jsonpickle.set_preferred_backend('simplejson')
            jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
            object = jsonpickle.decode(f.read())
        return object["test-container"]["test-container"]
