import pathlib
from typing import Dict

import docker
import luigi
import netaddr
from luigi import LocalTarget

from build_utils.lib.build_config import build_config
from build_utils.lib.build_or_pull_db_test_image import BuildOrPullDBTestContainerImage
from build_utils.lib.data.container_info import ContainerInfo
from build_utils.lib.data.dependency_collector.dependency_container_info_collector import CONTAINER_INFO
from build_utils.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from build_utils.lib.data.docker_network_info import DockerNetworkInfo
from build_utils.lib.docker_config import docker_config
from build_utils.lib.test_runner.create_export_directory import CreateExportDirectory
from build_utils.stoppable_task import StoppableTask


class SpawnTestContainer(StoppableTask):
    environment_name = luigi.Parameter()
    test_container_name = luigi.Parameter()
    network_info_dict = luigi.DictParameter(significant=False)
    ip_address_index_in_subnet = luigi.IntParameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._build_config = build_config()
        if self.ip_address_index_in_subnet < 0:
            raise Exception(
                "ip_address_index_in_subnet needs to be greater than 0 got %s"
                % self.ip_address_index_in_subnet)
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._test_container_info_target = luigi.LocalTarget(
            "%s/info/environment/%s/test-container/%s/container_info"
            % (self._build_config.output_directory,
               self.environment_name,
               self.test_container_name))
        if self._test_container_info_target.exists():
            self._test_container_info_target.remove()

    def output(self):
        return {CONTAINER_INFO: self._test_container_info_target}

    def requires(self):
        return {"test_container_image": BuildOrPullDBTestContainerImage(),
                "export_directory": CreateExportDirectory()}

    def run_task(self):
        test_container_image_info = self.get_test_container_image_info(self.input())
        network_info = DockerNetworkInfo.from_dict(self.network_info_dict)
        subnet = netaddr.IPNetwork(network_info.subnet)
        ip_address = str(subnet[2 + self.ip_address_index_in_subnet])
        # A later task which uses the test_container needs the exported container,
        # but to access exported container from inside the test_container,
        # we need to mount the release directory into the test_container.
        exports_host_path = pathlib.Path(self.get_release_directory()).absolute()
        tests_host_path = pathlib.Path("./tests").absolute()
        test_container = \
            self._client.containers.create(
                image=test_container_image_info.complete_name,
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
        with self.output()[CONTAINER_INFO].open("w") as file:
            container_info = ContainerInfo(container_name=self.test_container_name,
                                           ip_address=ip_address,
                                           network_info=network_info)
            file.write(container_info.to_json())

    def get_release_directory(self):
        return pathlib.Path(self.input()["export_directory"].path).absolute().parent

    def get_test_container_image_info(self, input: Dict[str, Dict[str, LocalTarget]]):
        image_info_of_dependencies = \
            DependencyImageInfoCollector().get_from_dict_of_inputs(input)
        test_container_image_info = image_info_of_dependencies["test_container_image"]
        return test_container_image_info
