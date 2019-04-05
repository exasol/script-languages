import pathlib

import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.data.container_info import ContainerInfo
from build_utils.lib.data.dependency_collector.dependency_container_info_collector import CONTAINER_INFO
from build_utils.lib.data.docker_network_info import DockerNetworkInfo
from build_utils.lib.data.image_info import ImageInfo
from build_utils.lib.docker_config import docker_config


class SpawnTestContainer(luigi.Task):
    test_container_name = luigi.Parameter()
    db_test_image_info_dict = luigi.DictParameter(significant=False)
    network_info_dict = luigi.DictParameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._build_config = build_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._test_container_info_target = luigi.LocalTarget(
            "%s/test-runner/db-test/test-container/%s/info"
            % (self._build_config.ouput_directory,
               self.test_container_name))
        if self._test_container_info_target.exists():
            self._test_container_info_target.remove()

    def output(self):
        return {CONTAINER_INFO: self._test_container_info_target}

    def run(self):
        db_test_image_info = ImageInfo.from_dict(self.db_test_image_info_dict)
        network_info = DockerNetworkInfo.from_dict(self.network_info_dict)
        release_host_path = pathlib.Path(self._build_config.ouput_directory + "/releases").absolute()
        tests_host_path = pathlib.Path("./tests").absolute()
        test_container = \
            self._client.containers.run(
                image=db_test_image_info.complete_name,
                name=self.test_container_name,
                network=network_info.network_name,
                command="sleep infinity",
                detach=True,
                volumes={
                    release_host_path: {
                        "bind": "/releases",
                        "mode": "ro"
                    },
                    tests_host_path: {
                        "bind": "/tests_src",
                        "mode": "ro"
                    }
                })
        test_container.exec_run(cmd="cp -r /tests_src /tests")
        with self.output()[CONTAINER_INFO].open("w") as file:
            container_info = ContainerInfo(self.test_container_name, network_info=network_info)
            file.write(container_info.to_json())
